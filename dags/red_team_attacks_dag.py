"""Airflow DAG: Run single-prompt + generative red teaming and generate reports.

This DAG is intentionally lightweight at import time.
Heavy imports (OpenAI, LangChain, pandas, VectorStore, etc.) are done inside tasks.

Datasets used (defaults):
- src/datasets/moonshot_jailbreak_prompts.csv
- src/datasets/Prompt Attacks Dataset - single turn.csv
- src/datasets/red_team_scenarios.json

Outputs (default):
- reports/<run_id>/single_prompt_results.csv
- reports/<run_id>/single_prompt_flagged.csv
- reports/<run_id>/generative_results.json
- reports/<run_id>/summary.json

Notes:
- Prompts may contain unsafe content. Tasks avoid logging prompt text to Airflow logs.
- Mutation tools are best-effort; missing optional deps will cause that tool to be skipped.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.models import Variable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_on_sys_path() -> None:
    """Ensure repo root is importable for task subprocesses.

    Airflow task runners often execute with a working directory and sys.path that
    do not include the DAG's repository root, so `import attacks...` can fail.
    """

    import sys

    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _run_id(context: Dict[str, Any]) -> str:
    # Stable run folder name per DagRun
    dr = context.get("dag_run")
    if dr and getattr(dr, "run_id", None):
        # e.g. manual__2026-03-19T12:34:56+00:00
        return str(dr.run_id).replace(":", "-")
    ts = context.get("ts") or datetime.utcnow().isoformat()
    return str(ts).replace(":", "-")


def _ensure_reports_dir(run_id: str) -> Path:
    out_dir = PROJECT_ROOT / "reports" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _resolve_path(value: str) -> str:
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


def _load_env() -> None:
    # Load repo-root .env if present
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=Path(".env"), override=False)
    except Exception:
        # dotenv is optional at runtime if env vars are already injected by Airflow
        pass


class ApiChatbot:
    """Adapter that makes the FastAPI service look like a local chatbot.

    This is used so existing attack runners (which expect `.chat()` and `.clear_history()`)
    can be reused without importing heavy ML deps inside Airflow tasks.
    """

    def __init__(self, base_url: str):
        self.base_url = (base_url or "").rstrip("/")

    def clear_history(self) -> None:
        import requests

        if not self.base_url:
            raise ValueError("CHAT_API_BASE_URL is not set")

        # Best-effort: if reset isn't available, ignore.
        try:
            requests.post(f"{self.base_url}/reset", timeout=30)
        except Exception:
            return

    def chat(self, message: str, debug: bool = False) -> str:
        import requests

        if not self.base_url:
            raise ValueError("CHAT_API_BASE_URL is not set")

        payload = {"message": message, "debug": bool(debug)}
        resp = requests.post(f"{self.base_url}/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "response" not in data:
            raise ValueError("Unexpected /chat response shape")
        return str(data["response"])


def _safe_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _load_mutation_tools(enabled: Optional[List[str]] = None):
    """Best-effort tool loading.

    Returns a list of tool instances with `.name` and `.apply(prompt)`.
    """

    enabled = enabled or [
        "char_swap",
        "homoglyph",
        "insert_punctuation",
        "payload_mask",
        "text_bugger",
        "text_fooler",
    ]

    tools = []

    _ensure_project_on_sys_path()

    def try_add(module_name: str, cls_name: str):
        try:
            mod = __import__(f"attacks.{module_name}", fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            tools.append(cls())
        except Exception:
            return

    mapping = {
        "char_swap": ("char_swap", "CharSwapTool"),
        "homoglyph": ("homoglyph", "HomoglyphTool"),
        "insert_punctuation": ("insert_punctuation", "InsertPunctuationTool"),
        "payload_mask": ("payload_mask", "PayloadMaskTool"),
        "text_bugger": ("text_bugger", "TextBuggerTool"),
        "text_fooler": ("text_fooler", "TextFoolerTool"),
    }

    for key in enabled:
        if key in mapping:
            try_add(*mapping[key])

    return tools


def task_run_single_prompt_attacks(**context):
    """Runs:
    1) Moonshot jailbreak prompts (as-is)
    2) Prompt Attacks single-turn dataset with per-tool iterative mutations

    Writes a single CSV of all attempts + a flagged-only CSV.
    """

    # Ensure relative paths resolve against repo root
    import os

    os.chdir(PROJECT_ROOT)
    _ensure_project_on_sys_path()

    run_id = _run_id(context)
    out_dir = _ensure_reports_dir(run_id)

    # Configurable via Airflow Variables
    chat_api_base_url = Variable.get("CHAT_API_BASE_URL", default_var="http://127.0.0.1:8000")

    moonshot_csv = Variable.get(
        "MOONSHOT_DATASET_PATH",
        default_var="src/datasets/moonshot_jailbreak_prompts.csv",
    )
    prompt_attacks_csv = Variable.get(
        "PROMPT_ATTACKS_DATASET_PATH",
        default_var="src/datasets/Prompt Attacks Dataset - single turn.csv",
    )

    enabled_tools_raw = Variable.get("PROMPT_MUTATION_TOOLS", default_var="")
    enabled_tools = [t.strip() for t in enabled_tools_raw.split(",") if t.strip()] or None

    chatbot = ApiChatbot(base_url=chat_api_base_url)

    import pandas as pd

    # Load datasets
    df_moonshot = pd.read_csv(_resolve_path(moonshot_csv))
    df_prompt_attacks = pd.read_csv(_resolve_path(prompt_attacks_csv))

    # Normalize columns
    for df in (df_moonshot, df_prompt_attacks):
        if "input" not in df.columns or "target" not in df.columns:
            raise ValueError("Dataset must have columns: input,target")

    tools = _load_mutation_tools(enabled_tools)

    rows: List[Dict[str, Any]] = []

    def eval_one(dataset: str, prompt: str, expected_refusal: str, variant: str, tool_name: str):
        # Avoid printing prompt text to logs; keep it only in saved artifacts.
        # Each single-turn attempt should be isolated.
        chatbot.clear_history()
        response = chatbot.chat(prompt)
        passed_safety = expected_refusal.strip().lower() in str(response).strip().lower()
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "tool": tool_name,
                "prompt_hash": _safe_hash(prompt),
                "prompt": prompt,
                "expected_refusal": expected_refusal,
                "actual_response": response,
                "passed_safety": bool(passed_safety),
            }
        )

    # 1) Moonshot: run prompts as-is
    for _, r in df_moonshot.iterrows():
        eval_one(
            dataset="moonshot_jailbreak_prompts",
            prompt=str(r["input"]),
            expected_refusal=str(r["target"]),
            variant="original",
            tool_name="none",
        )

    # 2) Prompt Attacks Dataset: original + iterative tool mutations
    for _, r in df_prompt_attacks.iterrows():
        original = str(r["input"])
        expected = str(r["target"])

        eval_one(
            dataset="prompt_attacks_single_turn",
            prompt=original,
            expected_refusal=expected,
            variant="original",
            tool_name="none",
        )

        for tool in tools:
            mutated = tool.apply(original)
            eval_one(
                dataset="prompt_attacks_single_turn",
                prompt=mutated,
                expected_refusal=expected,
                variant="mutated",
                tool_name=getattr(tool, "name", tool.__class__.__name__),
            )

    results_df = pd.DataFrame(rows)

    results_path = out_dir / "single_prompt_results.csv"
    results_df.to_csv(results_path, index=False)

    flagged = results_df[results_df["passed_safety"] == False].copy()  # noqa: E712
    flagged_path = out_dir / "single_prompt_flagged.csv"
    flagged.to_csv(flagged_path, index=False)

    summary = {
        "total_attempts": int(len(results_df)),
        "successful_jailbreaks": int(len(flagged)),
        "penetration_rate": (float(len(flagged)) / float(len(results_df)) * 100.0) if len(results_df) else 0.0,
        "results_csv": str(results_path),
        "flagged_csv": str(flagged_path),
        "tools_used": [getattr(t, "name", t.__class__.__name__) for t in tools],
    }

    summary_path = out_dir / "single_prompt_summary.json"
    summary_path.write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "results_csv": str(results_path),
        "flagged_csv": str(flagged_path),
        "summary_json": str(summary_path),
    }


def task_run_generative_red_team(**context):
    """Runs generative red teaming scenarios and writes JSON results into reports/<run_id>.

    This uses attacks/generative_red_team.py as-is.
    """

    import os

    os.chdir(PROJECT_ROOT)
    _ensure_project_on_sys_path()

    run_id = _run_id(context)
    out_dir = _ensure_reports_dir(run_id)

    chat_api_base_url = Variable.get("CHAT_API_BASE_URL", default_var="http://127.0.0.1:8000")

    # Make base URL available to attack modules (used for /search retrieval).
    os.environ["CHAT_API_BASE_URL"] = chat_api_base_url

    scenarios_path = Variable.get(
        "RED_TEAM_SCENARIOS_PATH",
        default_var="src/datasets/red_team_scenarios.json",
    )
    max_turns = int(Variable.get("GENERATIVE_MAX_TURNS", default_var="4"))

    chatbot = ApiChatbot(base_url=chat_api_base_url)

    from attacks.generative_red_team import run_generative_attack

    # Run generative attack; it writes to repo-root by default.
    run_generative_attack(chatbot, json_path=_resolve_path(scenarios_path), max_turns=max_turns, attack_tools=None)

    src = Path("generative_attack_evaluation_results.json")
    if not src.exists():
        raise FileNotFoundError("Expected generative_attack_evaluation_results.json not found")

    dst = out_dir / "generative_results.json"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "run_id": run_id,
        "generative_results_json": str(dst),
    }


def task_generate_reports(**context):
    """Combine single prompt + generative results into a single summary.json."""

    import os

    os.chdir(PROJECT_ROOT)
    _ensure_project_on_sys_path()

    run_id = _run_id(context)
    out_dir = _ensure_reports_dir(run_id)

    ti = context["ti"]
    single = ti.xcom_pull(task_ids="run_single_prompt_attacks") or {}
    gen = ti.xcom_pull(task_ids="run_generative_red_team") or {}

    import json
    import pandas as pd

    # Single prompt summary
    single_summary = {}
    single_flagged_rows = []
    if single.get("summary_json") and Path(single["summary_json"]).exists():
        single_summary = json.loads(Path(single["summary_json"]).read_text(encoding="utf-8"))
    if single.get("flagged_csv") and Path(single["flagged_csv"]).exists():
        df_flagged = pd.read_csv(single["flagged_csv"])
        # Keep flagged cases compact (don’t duplicate full prompt text in summary)
        single_flagged_rows = df_flagged[["dataset", "variant", "tool", "prompt_hash"]].to_dict("records")

    # Generative summary
    gen_summary = {}
    gen_flagged_rows = []
    gen_path = gen.get("generative_results_json")
    if gen_path and Path(gen_path).exists():
        data = json.loads(Path(gen_path).read_text(encoding="utf-8"))
        total = len(data)
        successes = sum(1 for r in data if r.get("was_jailbroken"))
        rate = (successes / total * 100.0) if total else 0.0
        gen_summary = {
            "total_scenarios": total,
            "successful_jailbreaks": successes,
            "penetration_rate": rate,
            "results_json": gen_path,
        }
        gen_flagged_rows = [
            {
                "scenario_id": r.get("scenario_id"),
                "attack_type": r.get("attack_type"),
                "turns_taken": r.get("turns_taken"),
            }
            for r in data
            if r.get("was_jailbroken")
        ]

    combined = {
        "run_id": run_id,
        "single_prompt": single_summary,
        "single_prompt_flagged_cases": single_flagged_rows,
        "generative": gen_summary,
        "generative_flagged_cases": gen_flagged_rows,
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    return {"run_id": run_id, "summary_json": str(out_path)}


with DAG(
    dag_id="sgbank_red_teaming",
    description="Run moonshot + prompt attacks + generative red teaming and report penetration rate.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"owner": "redteam"},
    tags=["redteam"],
) as dag:

    run_single_prompt_attacks = PythonOperator(
        task_id="run_single_prompt_attacks",
        python_callable=task_run_single_prompt_attacks,
    )

    run_generative_red_team = PythonOperator(
        task_id="run_generative_red_team",
        python_callable=task_run_generative_red_team,
    )

    generate_reports = PythonOperator(
        task_id="generate_reports",
        python_callable=task_generate_reports,
    )

    # When targeting a single shared FastAPI instance (api.py), serialize the tasks so they
    # don't fight over server-side chat history.
    run_single_prompt_attacks >> run_generative_red_team >> generate_reports

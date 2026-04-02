from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
import pandas as pd

from evaluator import (
    DEFAULT_EVALUATOR_CONDITION,
    DEFAULT_REFUSAL_SUGGESTION,
    llm_evaluate_jailbreak,
    passed_single_turn_safety,
)


CUSTOM_SINGLE_TURN_DATASET_FILENAME = "custom_single_turn_prompts.csv"
CUSTOM_MULTI_TURN_SCENARIOS_FILENAME = "custom_multi_turn_scenarios.json"


def _default_dataset_dirs() -> List[Path]:
    """Return dataset directories independent of current working directory.

    Streamlit is commonly launched from the `attacks/` folder, while other
    entrypoints run from the repo root. Using paths relative to this file makes
    discovery work in both cases.
    """

    here = Path(__file__).resolve().parent  # .../attacks
    repo_root = here.parent

    candidates: List[Path] = [
        here / "datasets",
        repo_root / "src" / "datasets",
        Path.cwd() / "attacks" / "datasets",
        Path.cwd() / "src" / "datasets",
    ]

    unique: List[Path] = []
    seen: set[str] = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rp)

    return unique


def discover_single_turn_dataset_paths(
    *,
    search_dirs: Optional[Sequence[Path]] = None,
) -> Dict[str, str]:
    """Return available single-turn CSV datasets the user can select.

    Conventions:
    - Custom prompts dataset: `custom_single_turn_prompts.csv`
    - General prompts datasets: any other `*.csv` files

    The mapping keys are display-friendly names; values are file paths.
    """

    dirs = list(search_dirs) if search_dirs is not None else _default_dataset_dirs()
    existing = [d for d in dirs if d.exists() and d.is_dir()]

    out: Dict[str, str] = {}

    # Custom dataset first (if present)
    for d in existing:
        p = d / CUSTOM_SINGLE_TURN_DATASET_FILENAME
        if p.exists() and p.is_file():
            out["custom_prompts"] = str(p)
            break

    # General datasets
    general: List[Path] = []
    for d in existing:
        for p in d.glob("*.csv"):
            if p.name == CUSTOM_SINGLE_TURN_DATASET_FILENAME:
                continue
            general.append(p)

    for p in sorted({p.resolve() for p in general}, key=lambda x: x.name.lower()):
        out[p.stem] = str(p)

    return out


def discover_multi_turn_scenarios_paths(
    *,
    search_dirs: Optional[Sequence[Path]] = None,
) -> Dict[str, str]:
    """Return available multi-turn scenario JSON files the user can select.

    Convention:
    - Custom multi-turn scenarios: `custom_multi_turn_scenarios.json`
    - Fallback: `red_team_scenarios.json`
    """

    dirs = list(search_dirs) if search_dirs is not None else _default_dataset_dirs()
    existing = [d for d in dirs if d.exists() and d.is_dir()]

    candidates: List[Tuple[str, Path]] = []
    for d in existing:
        for name, label in [
            (CUSTOM_MULTI_TURN_SCENARIOS_FILENAME, "custom_multi_turn"),
            ("red_team_scenarios.json", "red_team_scenarios"),
        ]:
            p = d / name
            if p.exists() and p.is_file():
                candidates.append((label, p))

    # Prefer custom first if it exists.
    ordered: Dict[str, str] = {}
    for label, path in sorted(candidates, key=lambda t: (0 if t[0] == "custom_multi_turn" else 1, t[1].name.lower())):
        if label not in ordered:
            ordered[label] = str(path)
    return ordered


def load_multi_turn_scenarios(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Scenarios JSON must be a list")
    return data


@dataclass
class ApiConfig:
    base_url: str
    chat_path: str = "/chat"
    reset_path: str = "/reset"
    timeout_s: float = 60.0

    # Request/response shape
    prompt_field: str = "message"
    response_field: str = "response"
    debug_field: Optional[str] = "debug"


class ChatbotApiClient:
    def __init__(self, config: ApiConfig):
        self.config = config

    def _url(self, path: str) -> str:
        return self.config.base_url.rstrip("/") + "/" + path.lstrip("/")

    async def reset(self, client: httpx.AsyncClient) -> None:
        try:
            await client.post(self._url(self.config.reset_path), timeout=self.config.timeout_s)
        except Exception:
            return

    async def chat(self, client: httpx.AsyncClient, prompt: str, debug: bool = False) -> str:
        payload: Dict[str, Any] = {self.config.prompt_field: prompt}
        if self.config.debug_field:
            payload[self.config.debug_field] = bool(debug)

        resp = await client.post(
            self._url(self.config.chat_path),
            json=payload,
            timeout=self.config.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or self.config.response_field not in data:
            raise ValueError("Unexpected chatbot response shape")
        return str(data[self.config.response_field])


def _backoff_s(attempt: int) -> float:
    base = min(8.0, 0.5 * (2 ** max(0, attempt - 1)))
    return base + random.random() * 0.25


async def _with_retries(fn, *, max_attempts: int = 5, retry_statuses: Sequence[int] = (429, 500, 502, 503, 504)):
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except httpx.HTTPStatusError as e:
            last_err = e
            status = e.response.status_code
            if status not in retry_statuses or attempt == max_attempts:
                raise
            await asyncio.sleep(_backoff_s(attempt))
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_err = e
            if attempt == max_attempts:
                raise
            await asyncio.sleep(_backoff_s(attempt))

    if last_err:
        raise last_err


def load_csv_prompts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "input" not in df.columns:
        raise ValueError("CSV must contain an 'input' column")
    if "target" not in df.columns:
        df["target"] = DEFAULT_REFUSAL_SUGGESTION
    return df


def load_tools(tool_keys: Optional[List[str]] = None) -> List[Any]:
    """Best-effort load of mutation tools under attacks/modules.

    Returns instantiated tools with `.apply(prompt)`.
    """

    keys = tool_keys or [
        "char_swap",
        "homoglyph",
        "insert_punctuation",
        "payload_mask",
        "text_bugger",
        "text_fooler",
    ]

    mapping: Dict[str, Tuple[str, str]] = {
        "char_swap": ("char_swap", "CharSwapTool"),
        "homoglyph": ("homoglyph", "HomoglyphTool"),
        "insert_punctuation": ("insert_punctuation", "InsertPunctuationTool"),
        "payload_mask": ("payload_mask", "PayloadMaskTool"),
        "text_bugger": ("text_bugger", "TextBuggerTool"),
        "text_fooler": ("text_fooler", "TextFoolerTool"),
    }

    tools: List[Any] = []
    for k in keys:
        if k not in mapping:
            continue
        mod_name, cls_name = mapping[k]
        try:
            mod = __import__(f"modules.{mod_name}", fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            tools.append(cls())
        except Exception:
            continue

    return tools


def apply_tools(prompt: str, tools: Sequence[Any]) -> str:
    mutated = prompt
    for t in tools:
        try:
            mutated = t.apply(mutated)
        except Exception:
            continue
    return mutated


async def run_single_turn(
    *,
    api: ApiConfig,
    prompts_df: pd.DataFrame,
    tools: Sequence[Any] = (),
    refusal_suggestion: str = DEFAULT_REFUSAL_SUGGESTION,
    concurrency: int = 6,
    debug: bool = False,
) -> pd.DataFrame:
    """Run single-turn attacks.

    Output columns: dataset, category, attack_type, variant, tool, input, expected_refusal, actual_response, passed_safety
    """

    client = ChatbotApiClient(api)
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    # Normalize metadata columns
    df = prompts_df.copy()
    if "category" not in df.columns:
        df["category"] = "uncategorized"
    if "attack_type" not in df.columns:
        df["attack_type"] = "single_turn"
    if "dataset" not in df.columns:
        df["dataset"] = "custom"
    if "target" not in df.columns:
        df["target"] = refusal_suggestion

    rows: List[Dict[str, Any]] = []

    async with httpx.AsyncClient() as http:

        async def eval_one(prompt_text: str, expected: str, meta: Dict[str, Any]) -> None:
            async with sem:
                await client.reset(http)

                async def do_chat():
                    return await client.chat(http, prompt_text, debug=debug)

                response = await _with_retries(do_chat)

                passed = passed_single_turn_safety(
                    response_text=response,
                    refusal_suggestion=str(expected or refusal_suggestion),
                )

                rows.append(
                    {
                        **meta,
                        "input": prompt_text,
                        "expected_refusal": str(expected or refusal_suggestion),
                        "actual_response": response,
                        "passed_safety": bool(passed),
                    }
                )

        tasks: List[asyncio.Task] = []

        for _, r in df.iterrows():
            original = str(r.get("input", ""))
            expected = str(r.get("target") or refusal_suggestion)

            base_meta = {
                "dataset": str(r.get("dataset")),
                "category": str(r.get("category")),
                "attack_type": str(r.get("attack_type")),
            }

            tasks.append(asyncio.create_task(eval_one(original, expected, {**base_meta, "variant": "original", "tool": "none"})))

            for t in tools:
                tool_name = getattr(t, "name", t.__class__.__name__)
                mutated = apply_tools(original, [t])
                tasks.append(
                    asyncio.create_task(
                        eval_one(mutated, expected, {**base_meta, "variant": "mutated", "tool": str(tool_name)})
                    )
                )

        await asyncio.gather(*tasks)

    return pd.DataFrame(rows)


async def run_multi_turn_scenarios(
    *,
    api: ApiConfig,
    scenarios: List[Dict[str, Any]],
    evaluator_condition: str = DEFAULT_EVALUATOR_CONDITION,
    custom_prompt: Optional[str] = None,
    tools: Sequence[Any] = (),
    max_turns: int = 6,
    concurrency: int = 2,
    sequential: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """Run multi-turn scenarios.

    Returns a scenario-level DataFrame with was_jailbroken and metadata.
    """

    client = ChatbotApiClient(api)
    rows: List[Dict[str, Any]] = []

    async with httpx.AsyncClient() as http:

        async def run_one(sc: Dict[str, Any]) -> None:
            await client.reset(http)

            from generative_red_team import RedTeamAttacker

            attacker = RedTeamAttacker(
                model="gpt-4o-mini",
                tools=list(tools) if tools else None,
                custom_prompt=custom_prompt,
            )

            scenario_id = sc.get("scenario_id")
            attack_type = sc.get("attack_type")
            objective = sc.get("description") or sc.get("objective") or ""
            turns = sc.get("turns") or []
            if not isinstance(turns, list) or not turns:
                return

            history = ""
            log: List[Dict[str, Any]] = []
            jailbroken = False

            seed = str(turns[0].get("input") or "")
            current_prompt = apply_tools(seed, tools) if tools else seed

            for turn_idx in range(int(max_turns)):

                async def do_chat():
                    return await client.chat(http, current_prompt, debug=debug)

                response = await _with_retries(do_chat)

                eval_res = llm_evaluate_jailbreak(
                    prompt=current_prompt,
                    response_text=response,
                    evaluator_condition=evaluator_condition,
                )

                log.append(
                    {
                        "turn": turn_idx + 1,
                        "input": current_prompt,
                        "response": response,
                        "eval_success": bool(eval_res.success),
                        "eval_categories": eval_res.categories,
                        "eval_is_summary_like": eval_res.is_summary_like,
                    }
                )

                history += f"User: {current_prompt}\nBot: {response}\n\n"

                if eval_res.success:
                    jailbroken = True
                    break

                # Suggested reference prompt, if present.
                ref_idx = min(turn_idx + 1, len(turns) - 1)
                prompt_reference = str(turns[ref_idx].get("input") or turns[0].get("input") or "")
                current_prompt = attacker.generate_next_attack(str(objective), history, prompt_reference)

            rows.append(
                {
                    "dataset": "custom_multiturn_scenario",
                    "category": "custom_multiturn_scenario",
                    "attack_type": str(attack_type or "scenario"),
                    "scenario_id": scenario_id,
                    "objective": objective,
                    "was_jailbroken": bool(jailbroken),
                    "turns_taken": int(len(log)),
                    "log": json.dumps(log, ensure_ascii=False),
                }
            )

        if sequential:
            for sc in scenarios:
                await run_one(sc)
        else:
            sem = asyncio.Semaphore(max(1, int(concurrency)))

            async def run_one_limited(sc: Dict[str, Any]) -> None:
                async with sem:
                    await run_one(sc)

            await asyncio.gather(*[asyncio.create_task(run_one_limited(sc)) for sc in scenarios])

    return pd.DataFrame(rows)


def ensure_reports_dir() -> Path:
    here = Path(__file__).resolve().parent
    out = here / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out

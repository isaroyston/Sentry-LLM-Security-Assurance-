from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

from env_utils import load_env
from evaluator import DEFAULT_EVALUATOR_CONDITION, DEFAULT_REFUSAL_SUGGESTION
from generative_red_team import DEFAULT_WITHDRAWAL_CUSTOM_PROMPT
from red_teaming import (
    ApiConfig,
    discover_multi_turn_scenarios_paths,
    discover_single_turn_dataset_paths,
    ensure_reports_dir,
    load_csv_prompts,
    load_tools,
    run_multi_turn_scenarios,
    run_single_turn,
)


def _safe_read_json(path: str) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _donut(df: pd.DataFrame, names: str, values: str, title: str):
    if px is None:
        st.warning("Plotly Express not installed; skipping chart.")
        return

    # Map labels to colors
    color_map = {
        "safe": "green",
        "passed": "green",
        "jailbroken": "red",
        "failed": "red",
    }

    fig = px.pie(
        df, names=names, values=values, hole=0.6, title=title, color=names, color_discrete_map=color_map
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def _compute_overall_metrics(results: pd.DataFrame) -> Tuple[int, int, float]:
    total = int(len(results))
    if total == 0:
        return 0, 0, 0.0
    failed = int((results["passed_safety"] == False).sum())  # noqa: E712
    rate = failed / total * 100.0
    return total, failed, rate


def _compute_category_breakdown(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["category", "total", "failed", "fail_rate"])
    g = results.groupby("category", dropna=False)
    out = g.agg(
        total=("passed_safety", "size"),
        failed=("passed_safety", lambda s: int((s == False).sum())),  # noqa: E712
    ).reset_index()
    out["fail_rate"] = out.apply(lambda r: (r["failed"] / r["total"] * 100.0) if r["total"] else 0.0, axis=1)
    out = out.sort_values(["failed", "total"], ascending=False)
    return out


def _vulnerability_checklist(results: pd.DataFrame) -> pd.DataFrame:
    """A simple checklist: category is 'defended' if no failed cases in that category."""

    if results.empty:
        return pd.DataFrame(columns=["category", "defended"])

    breakdown = _compute_category_breakdown(results)
    breakdown["defended"] = breakdown["failed"].apply(lambda x: x == 0)
    return breakdown[["category", "defended"]]


st.set_page_config(page_title="Red Team Dashboard", layout="wide")

# Persist run outputs so the dashboard doesn't disappear on widget changes.
if "report_df" not in st.session_state:
    st.session_state["report_df"] = None
if "report_run_id" not in st.session_state:
    st.session_state["report_run_id"] = None
if "single_results" not in st.session_state:
    st.session_state["single_results"] = None
if "multi_results" not in st.session_state:
    st.session_state["multi_results"] = None

# Load env (attacks/.env preferred, then repo-root .env)
load_env(override=False)

st.title("Red Team Dashboard")
st.markdown(
    """
    <style>
      section[data-testid="stSidebar"] { width: 420px !important; }
      section[data-testid="stSidebar"] > div { width: 420px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Dataset discovery (used by sidebar controls)
# -------------------------

single_turn_paths = discover_single_turn_dataset_paths()
generic_dataset_options = [k for k in single_turn_paths.keys() if k != "custom_prompts"]

builtin_custom_single_turn_path = single_turn_paths.get("custom_prompts")

scenario_sources = discover_multi_turn_scenarios_paths()

locked = st.session_state.get("report_df") is not None

with st.sidebar:
    st.header("1) Connect Chatbot API")

    if locked:
        st.caption("Results are shown below. Clear results to run again.")
        if st.button("Clear Results"):
            st.session_state["report_df"] = None
            st.session_state["report_run_id"] = None
            st.session_state["single_results"] = None
            st.session_state["multi_results"] = None
            st.rerun()

    base_url = st.text_input(
        "Base URL",
        value=os.getenv("CHAT_API_BASE_URL", "http://127.0.0.1:8000"),
        disabled=locked,
    )
    chat_path = st.text_input("Chat path", value="/chat", disabled=locked)
    reset_path = st.text_input("Reset path", value="/reset", disabled=locked)

    st.caption("If your API uses a different JSON shape, adjust these fields.")
    prompt_field = st.text_input("Request prompt field", value="message", disabled=locked)
    response_field = st.text_input("Response field", value="response", disabled=locked)
    debug_field = st.text_input("Debug field (optional)", value="debug", disabled=locked)
    debug_enabled = st.checkbox("Send debug flag", value=False, disabled=locked)

    st.header("2) Select Attack Modes")
    modes = st.multiselect(
        "Run",
        options=[
            "General (static datasets)",
            "Custom Single-turn Prompts",
            "Red Team LLM (multi-turn scenarios)",
        ],
        default=["General (static datasets)"],
        disabled=locked,
    )

    st.header("3) Configure Attacks")

    # --- General Datasets ---
    selected_generic_datasets: List[str] = []
    if "General (static datasets)" in modes:
        with st.expander("General Datasets", expanded=True):
            if generic_dataset_options:
                selected_generic_datasets = st.multiselect(
                    "Select datasets",
                    options=generic_dataset_options,
                    default=list(generic_dataset_options),
                    disabled=locked,
                )
            else:
                st.warning(
                    "No generic datasets found. Expected `*.csv` under `attacks/datasets/` (or `src/datasets/`).\n\n"
                    f"Discovered keys: {list(single_turn_paths.keys())}"
                )

    # --- Custom Single-turn ---
    custom_single_turn_source = None
    if "Custom Single-turn Prompts" in modes:
        with st.expander("Custom Single-turn Prompts", expanded=True):
            source_options = []
            if builtin_custom_single_turn_path:
                source_options.append("Built-in (custom_single_turn_prompts.csv)")
            source_options += ["Upload CSV", "Editable Table"]
            custom_single_turn_source = st.radio(
                "Source",
                options=source_options,
                index=0,
                horizontal=True,
                disabled=locked,
            )

    # --- Multi-turn Scenarios ---
    scenario_source_key = None
    if "Red Team LLM (multi-turn scenarios)" in modes:
        with st.expander("Multi-turn Scenarios", expanded=True):
            if scenario_sources:
                scenario_source_key = st.selectbox(
                    "Select scenario source",
                    options=list(scenario_sources.keys()),
                    index=0,
                    help="Uses `custom_multi_turn_scenarios.json` if found, else falls back to `red_team_scenarios.json`.",
                    disabled=locked,
                )
            else:
                st.warning(
                    "No multi-turn scenario files found. Expected `custom_multi_turn_scenarios.json` under `attacks/datasets/` (or `src/datasets/`)."
                )

    st.header("4) Prompt Mutation Tools")
    tool_options = ["char_swap", "homoglyph", "insert_punctuation", "payload_mask", "text_bugger", "text_fooler"]
    selected_tools = st.multiselect("Tools", options=tool_options, default=[], disabled=locked)

    st.header("5) Evaluator Settings")
    refusal_suggestion = st.text_area(
        "Refusal suggestion (single-turn)",
        value=os.getenv("REFUSAL_SUGGESTION", DEFAULT_REFUSAL_SUGGESTION),
        height=80,
        disabled=locked,
    )
    evaluator_condition = st.text_area(
        "LLM evaluator condition (multi-turn)",
        value=DEFAULT_EVALUATOR_CONDITION,
        height=140,
        disabled=locked,
    )

    st.header("6) Attacker Prompt (multi-turn)")
    if "attacker_prompt_text" not in st.session_state:
        st.session_state["attacker_prompt_text"] = DEFAULT_WITHDRAWAL_CUSTOM_PROMPT

    attacker_custom_prompt = st.text_area(
        "Attacker system prompt (editable)",
        value=str(st.session_state["attacker_prompt_text"]),
        height=200,
        help="This is passed into the attacker as domain/context instructions.",
        disabled=locked,
    )
    st.session_state["attacker_prompt_text"] = attacker_custom_prompt

    st.header("7) Execution")
    concurrency = 6
    max_turns = st.slider("Max turns (multi-turn)", min_value=1, max_value=10, value=6, disabled=locked)

    run_btn = st.button("Run Red Teaming", type="primary", disabled=locked)

api_cfg = ApiConfig(
    base_url=base_url,
    chat_path=chat_path,
    reset_path=reset_path,
    prompt_field=prompt_field,
    response_field=response_field,
    debug_field=(debug_field.strip() or None),
)

tools = load_tools(selected_tools) if selected_tools else []

custom_df = None
# Hide the pre-run editors after a run; dashboard should not change due to sidebar edits.
if (not locked) and ("Custom Single-turn Prompts" in modes):
    if custom_single_turn_source == "Built-in (custom_single_turn_prompts.csv)":
        st.subheader("Built-in Custom Prompts")
        if not builtin_custom_single_turn_path:
            st.error("Built-in custom prompts file not found.")
        else:
            try:
                custom_df = load_csv_prompts(builtin_custom_single_turn_path)
                st.caption(f"Loaded from `{Path(builtin_custom_single_turn_path).name}`")
                custom_df = st.data_editor(
                    custom_df,
                    use_container_width=True,
                    num_rows="dynamic",
                    column_config={
                        "input": st.column_config.TextColumn("input", width="large"),
                        "target": st.column_config.TextColumn("target", width="large"),
                    },
                )
            except Exception as e:
                st.error(f"Failed to load built-in custom prompts: {e}")
                custom_df = None

    elif custom_single_turn_source == "Upload CSV":
        st.subheader("Upload Custom Prompts")
        st.info("CSV must contain `input` and `target` columns.")
        uploaded_file = st.file_uploader("Choose a file...", type=["csv"])
        if uploaded_file:
            try:
                custom_df = pd.read_csv(uploaded_file)
                if "input" not in custom_df.columns or "target" not in custom_df.columns:
                    st.error("Invalid CSV. Must contain `input` and `target` columns.")
                    custom_df = None
                else:
                    st.success(f"Loaded {len(custom_df)} prompts from `{uploaded_file.name}`.")
                    custom_df = st.data_editor(
                        custom_df,
                        use_container_width=True,
                        num_rows="dynamic",
                        column_config={
                            "input": st.column_config.TextColumn("input", width="large"),
                            "target": st.column_config.TextColumn("target", width="large"),
                        },
                    )
            except Exception as e:
                st.error(f"Error reading file: {e}")

    elif custom_single_turn_source == "Editable Table":
        st.subheader("Custom Prompts Table")
        st.caption("Edit prompts directly. Required columns: `input`, `target`.")

        default_rows = pd.DataFrame(
            {
                "input": ["" for _ in range(5)],
                "target": ["" for _ in range(5)],
            }
        )
        seed_df = st.session_state.get("custom_single_turn_table", default_rows)
        edited = st.data_editor(
            seed_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "input": st.column_config.TextColumn("input", width="large"),
                "target": st.column_config.TextColumn("target", width="large"),
            },
        )
        st.session_state["custom_single_turn_table"] = edited
        # Only use non-empty inputs
        cleaned = edited.copy()
        cleaned["input"] = cleaned["input"].astype(str).str.strip()
        cleaned["target"] = cleaned["target"].astype(str)
        cleaned = cleaned[cleaned["input"] != ""]
        if not cleaned.empty:
            custom_df = cleaned
            st.success(f"Loaded {len(custom_df)} prompts from table.")


scenario_json_text = None
if (not locked) and ("Red Team LLM (multi-turn scenarios)" in modes and scenario_source_key):
    st.subheader("Multi-turn Scenarios")
    path = scenario_sources[scenario_source_key]
    st.info(f"Using scenarios from `{Path(path).name}`.")
    try:
        default_text = Path(path).read_text(encoding="utf-8")

        state_key = f"scenario_json_text::{scenario_source_key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = default_text

        edited_text = st.text_area(
            "Scenarios JSON (editable)",
            value=str(st.session_state[state_key]),
            height=320,
            help="Must be a JSON list of scenario objects.",
        )
        st.session_state[state_key] = edited_text
        scenario_json_text = edited_text

        with st.expander("Preview parsed JSON", expanded=False):
            try:
                parsed = json.loads(scenario_json_text or "[]")
                if not isinstance(parsed, list):
                    st.error("Scenarios JSON must be a list (top-level array).")
                else:
                    st.json(parsed)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
    except Exception as e:
        st.error(f"Error loading scenarios from `{path}`: {e}")
        scenario_json_text = None

# -------------------------
# Run
# -------------------------
if run_btn:
    if not base_url.strip():
        st.error("Base URL is required")
        st.stop()

    out_dir = ensure_reports_dir()

    all_single: List[pd.DataFrame] = []

    single_turn_jobs: List[str] = []
    if "General (static datasets)" in modes and selected_generic_datasets:
        single_turn_jobs.extend([f"dataset:{k}" for k in selected_generic_datasets])
    if "Custom Single-turn Prompts" in modes and custom_df is not None:
        single_turn_jobs.append("custom_single_turn")

    single_progress = st.progress(0.0) if single_turn_jobs else None
    single_status = st.empty() if single_turn_jobs else None
    single_done = 0

    with st.spinner("Running single-turn attacks..."):
        if "General (static datasets)" in modes and selected_generic_datasets:
            for key in selected_generic_datasets:
                path = single_turn_paths.get(key)
                if not path:
                    continue

                df = load_csv_prompts(path)
                # Dataset naming/classification defaults
                df["dataset"] = Path(path).stem
                if "category" not in df.columns:
                    df["category"] = df.get("category", Path(path).stem)
                if "attack_type" not in df.columns:
                    df["attack_type"] = df.get("attack_type", "single_turn")

                if single_status is not None:
                    single_status.write(
                        f"Running single-turn dataset '{key}' with {len(tools)} tool(s)…"
                    )

                res = asyncio.run(
                    run_single_turn(
                        api=api_cfg,
                        prompts_df=df,
                        tools=tools,
                        refusal_suggestion=refusal_suggestion,
                        concurrency=concurrency,
                        debug=debug_enabled,
                    )
                )
                all_single.append(res)

                single_done += 1
                if single_progress is not None:
                    single_progress.progress(min(1.0, single_done / max(1, len(single_turn_jobs))))

        if "Custom Single-turn Prompts" in modes and custom_df is not None:
            df = custom_df.copy()
            df["dataset"] = "custom_single_turn"
            if "category" not in df.columns:
                df["category"] = "custom"
            if "attack_type" not in df.columns:
                df["attack_type"] = "single_turn"
            if "target" not in df.columns:
                df["target"] = ""

            if single_status is not None:
                single_status.write(
                    f"Running custom single-turn prompts with {len(tools)} tool(s)…"
                )
            res = asyncio.run(
                run_single_turn(
                    api=api_cfg,
                    prompts_df=df,
                    tools=tools,
                    refusal_suggestion=refusal_suggestion,
                    concurrency=concurrency,
                    debug=debug_enabled,
                )
            )
            all_single.append(res)

            single_done += 1
            if single_progress is not None:
                single_progress.progress(min(1.0, single_done / max(1, len(single_turn_jobs))))

    if single_status is not None:
        single_status.empty()

    single_results = pd.concat(all_single, ignore_index=True) if all_single else pd.DataFrame()

    multi_results = pd.DataFrame()
    with st.spinner("Running multi-turn scenarios..."):
        if "Red Team LLM (multi-turn scenarios)" in modes:
            try:
                scenarios = json.loads(scenario_json_text or "[]")
                if not isinstance(scenarios, list):
                    raise ValueError("Scenarios JSON must be a list")
            except Exception as e:
                st.error(f"Invalid scenarios JSON: {e}")
                scenarios = []

            if scenarios:
                st.caption(f"Running {len(scenarios)} scenario(s) sequentially…")
                multi_results = asyncio.run(
                    run_multi_turn_scenarios(
                        api=api_cfg,
                        scenarios=scenarios,
                        evaluator_condition=evaluator_condition,
                        custom_prompt=(attacker_custom_prompt.strip() or DEFAULT_WITHDRAWAL_CUSTOM_PROMPT),
                        tools=tools,
                        max_turns=max_turns,
                        sequential=True,
                        concurrency=1,
                        debug=debug_enabled,
                    )
                )

    # Persist outputs
    if not single_results.empty:
        single_path = out_dir / "single_turn_results.csv"
        single_results.to_csv(single_path, index=False)

    if not multi_results.empty:
        multi_path = out_dir / "multi_turn_results.csv"
        multi_results.to_csv(multi_path, index=False)

    # -------------------------
    # Dashboard
    # -------------------------
    st.divider()

    # Build and persist the dashboard data so UI interactions don't hide it.
    # --- Consolidate results ---
    all_results: List[Dict[str, Any]] = []
    if not single_results.empty:
        for _, r in single_results.iterrows():
            all_results.append(
                {
                    "run_type": "single_turn",
                    "dataset": r.get("dataset"),
                    "category": r.get("category"),
                    "attack_type": r.get("attack_type"),
                    "prompt": r.get("input"),
                    "response": r.get("actual_response"),
                    "passed_safety": r.get("passed_safety"),
                    "was_jailbroken": not bool(r.get("passed_safety")),
                    "details": f"tool={r.get('tool')}; variant={r.get('variant')}",
                }
            )

    if not multi_results.empty:
        for _, r in multi_results.iterrows():
            last_prompt = ""
            last_response = ""
            try:
                log = json.loads(str(r.get("log") or "[]"))
                if isinstance(log, list) and log:
                    last_prompt = str(log[-1].get("input") or "")
                    last_response = str(log[-1].get("response") or "")
            except Exception:
                pass

            all_results.append(
                {
                    "run_type": "multi_turn",
                    "dataset": r.get("dataset") or "custom_multiturn_scenario",
                    "category": r.get("category") or "custom_multiturn_scenario",
                    "attack_type": r.get("attack_type"),
                    "prompt": last_prompt or r.get("objective"),
                    "response": last_response,
                    "passed_safety": not bool(r.get("was_jailbroken")),
                    "was_jailbroken": bool(r.get("was_jailbroken")),
                    "details": f"scenario_id={r.get('scenario_id')}; turns_taken={r.get('turns_taken')}",
                }
            )

    if not all_results:
        st.warning("The run produced no results.")
        st.stop()

    report_df = pd.DataFrame(all_results)
    report_df["passed_safety"] = report_df["passed_safety"].astype(bool)
    # Normalize strings to prevent filter/sort issues with NaNs.
    for col, default in {
        "dataset": "unknown_dataset",
        "category": "uncategorized",
        "attack_type": "unknown_attack",
        "run_type": "unknown_run",
    }.items():
        report_df[col] = report_df[col].fillna(default).astype(str)

    run_id = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    report_path = ensure_reports_dir() / run_id
    report_df.to_csv(report_path, index=False)

    st.session_state["single_results"] = single_results
    st.session_state["multi_results"] = multi_results
    st.session_state["report_df"] = report_df
    st.session_state["report_run_id"] = run_id

    st.success("Run complete.")
    st.rerun()


# -------------------------
# Dashboard (from last run)
# -------------------------
report_df = st.session_state.get("report_df")
if report_df is not None:
    st.divider()
    st.header("Report")

    run_id = st.session_state.get("report_run_id") or "red_team_report.csv"
    st.download_button("Download Full Report (CSV)", data=report_df.to_csv(index=False), file_name=run_id)

    st.subheader("Overall Summary")
    total, failed, rate = _compute_overall_metrics(report_df)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Tests", f"{total}")
        st.metric("Jailbreaks", f"{failed}")
        st.metric("Jailbreak Rate", f"{rate:.2f}%")
    with c2:
        summary_chart_df = pd.DataFrame(
            {
                "status": ["safe", "jailbroken"],
                "count": [total - failed, failed],
            }
        )
        _donut(summary_chart_df, names="status", values="count", title="Overall Status")

    st.subheader("Breakdown by Category")
    available_datasets = ["All"] + sorted(report_df["dataset"].dropna().astype(str).unique().tolist())
    selected_dataset = st.selectbox("View dataset", options=available_datasets, key="report_dataset_filter")

    view_df = report_df
    if selected_dataset != "All":
        view_df = report_df[report_df["dataset"] == selected_dataset]

    if view_df.empty:
        st.warning("No data for the selected filter.")
        st.stop()

    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        category_breakdown = _compute_category_breakdown(view_df)
        st.dataframe(
            category_breakdown.style.format({"fail_rate": "{:.2f}%"}).bar(
                subset=["total", "failed"], color="#888", vmin=0
            ),
            use_container_width=True,
        )
    with c2:
        _donut(category_breakdown, names="category", values="failed", title="Jailbreaks by Category")

    st.subheader("Inspect Results")
    filter_cols = st.columns(3)
    run_type_opts = ["All"] + sorted(view_df["run_type"].dropna().astype(str).unique().tolist())
    category_opts = ["All"] + sorted(view_df["category"].dropna().astype(str).unique().tolist())
    run_type_filter = filter_cols[0].selectbox("Run Type", run_type_opts, key="report_run_type_filter")
    category_filter = filter_cols[1].selectbox("Category", category_opts, key="report_category_filter")
    passed_filter = filter_cols[2].selectbox("Result", ["All", "Jailbroken", "Safe"], key="report_passed_filter")

    inspection_df = view_df
    if run_type_filter != "All":
        inspection_df = inspection_df[inspection_df["run_type"] == run_type_filter]
    if category_filter != "All":
        inspection_df = inspection_df[inspection_df["category"] == category_filter]
    if passed_filter != "All":
        is_safe = passed_filter == "Safe"
        inspection_df = inspection_df[inspection_df["passed_safety"] == is_safe]

    st.data_editor(
        inspection_df[["dataset", "category", "attack_type", "prompt", "response", "passed_safety", "details"]],
        use_container_width=True,
        height=400,
        column_config={
            "passed_safety": st.column_config.CheckboxColumn("Safe?", default=False),
            "prompt": st.column_config.TextColumn(width="large"),
            "response": st.column_config.TextColumn(width="large"),
        },
        disabled=True,
    )

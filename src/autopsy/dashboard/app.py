"""Streamlit dashboard for Phases 3-4 — visualize and compare batch evaluation results.

Single-run mode (Phase 3):
    Load a saved BatchResultSet JSON and explore results, failure distribution, drill-down

Comparison mode (Phase 4):
    Load two BatchResultSet JSONs and compare baseline vs. experiment with diffs

History mode (Phase 4):
    Load a RunHistory JSON and track trends, detect regressions, view past runs

Usage:
    # Single run
    streamlit run src/autopsy/dashboard/app.py -- --mode single --data results.json

    # Compare two runs
    streamlit run src/autopsy/dashboard/app.py -- --mode compare --baseline baseline.json --current current.json

    # View history
    streamlit run src/autopsy/dashboard/app.py -- --mode history --history history.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

try:
    from ..eval.aggregation import BatchResultSet
    from ..eval.comparison import compare
    from ..eval.history import RunHistory
except ImportError:
    # When run as a Streamlit app, imports may need adjustment
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.autopsy.eval.aggregation import BatchResultSet
    from src.autopsy.eval.comparison import compare
    from src.autopsy.eval.history import RunHistory


def display_single_run(result_set: BatchResultSet):
    """Display a single batch result set (Phase 3 mode)."""
    # Header with summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cases", result_set.total_cases)
    with col2:
        st.metric("Passed", result_set.metrics.passed)
    with col3:
        st.metric("Failed", result_set.metrics.failed)
    with col4:
        st.metric("Errors", result_set.metrics.errors)

    st.metric("Pass Rate", f"{result_set.metrics.pass_rate:.1%}")

    # Visualizations
    st.header("📈 Results Overview")
    col1, col2 = st.columns(2)

    # Pass/fail chart
    with col1:
        st.subheader("Pass/Fail Breakdown")
        if result_set.total_cases > 0:
            data = {
                "Passed": result_set.metrics.passed,
                "Failed": result_set.metrics.failed,
                "Errors": result_set.metrics.errors,
            }
            st.bar_chart(data)

    # Failure distribution
    with col2:
        st.subheader("Failure Distribution")
        if result_set.metrics.failure_distribution:
            st.bar_chart(result_set.metrics.failure_distribution)
        else:
            st.info("No failures to display.")

    # Detailed results table
    st.header("🔍 Case-by-Case Results")
    if result_set.results:
        # Build a table for display
        table_data = []
        for case in result_set.results:
            status = "✅ Pass" if case.passed else "❌ Fail" if case.failed else "💥 Error"
            table_data.append(
                {
                    "Index": case.case_index,
                    "Status": status,
                    "Score": f"{case.overall_score:.2f}" if case.overall_score else "—",
                    "Root Cause": case.root_cause,
                    "Confidence": f"{case.confidence:.0%}" if case.confidence else "—",
                    "Failures": ", ".join(case.failure_labels) if case.failure_labels else "—",
                }
            )

        st.dataframe(table_data, use_container_width=True)

        # Drill-down: select a case to inspect
        st.subheader("📋 Drill-Down: Inspect a Case")
        case_idx = st.selectbox(
            "Select a case:",
            options=range(len(result_set.results)),
            format_func=lambda i: f"Case {i}: {result_set.results[i].root_cause}",
        )

        if case_idx is not None:
            case = result_set.results[case_idx]
            st.write(f"**Case {case_idx}** — Trace ID: `{case.trace_id}`")

            # Status & score
            col1, col2 = st.columns(2)
            with col1:
                if case.passed:
                    st.success("✅ PASSED")
                elif case.failed:
                    st.error("❌ FAILED")
                else:
                    st.warning("💥 ERROR")
            with col2:
                if case.overall_score is not None:
                    st.write(f"**Overall Score:** {case.overall_score:.3f}")

            # Diagnosis
            st.subheader("Diagnosis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Root Cause:** `{case.root_cause}`")
            with col2:
                confidence_str = f"{case.confidence:.0%}" if case.confidence else "—"
                st.write(f"**Confidence:** {confidence_str}")
            with col3:
                if case.culprit_span_id:
                    st.write(f"**Culprit Span:** `{case.culprit_span_id}`")

            st.write(f"**Rationale:** {case.diagnosis_rationale}")

            # Failure signals from evaluation
            if case.failure_labels:
                st.subheader("Failure Signals (from Phase 1 Evaluation)")
                st.write(", ".join(f"`{label}`" for label in case.failure_labels))

            # Eval results (metric-by-metric breakdown)
            if case.eval_results:
                st.subheader("Metric Results")
                metric_data = []
                for result in case.eval_results:
                    metric_data.append(
                        {
                            "Metric": result.get("name", "—"),
                            "Score": f"{result.get('score', 0):.2f}" if result.get("score") is not None else "—",
                            "Passed": "✓" if result.get("passed") else "✗" if result.get("passed") is False else "—",
                            "Failure Label": result.get("failure_label", "—"),
                            "Rationale": (result.get("rationale", "—")[:60] + "…"),
                        }
                    )
                st.dataframe(metric_data, use_container_width=True)


def display_comparison(baseline_set: BatchResultSet, current_set: BatchResultSet):
    """Display a comparison of two batch result sets (Phase 4 mode)."""
    st.header("📊 Comparison: Baseline vs. Current")

    # Compute comparison
    comparison = compare(baseline_set, current_set)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Baseline Pass Rate", f"{comparison.baseline_pass_rate:.1%}")
    with col2:
        st.metric("Current Pass Rate", f"{comparison.current_pass_rate:.1%}")
    with col3:
        delta_str = f"{comparison.pass_rate_delta:+.1%}"
        delta_color = "🟢" if comparison.pass_rate_delta > 0 else "🔴" if comparison.pass_rate_delta < 0 else "⚪"
        st.metric("Pass Rate Delta", f"{delta_color} {delta_str}")
    with col4:
        improvement_str = f"↑ {comparison.improvement_count}"
        regression_str = f"↓ {comparison.regression_count}"
        status = f"{improvement_str} / {regression_str}"
        st.metric("Improvements / Regressions", status)

    # Details
    st.subheader("Summary")
    if comparison.improved:
        st.success(f"✓ **{comparison.improvement_count} case(s) improved**")
    if comparison.regression:
        st.error(f"✗ **{comparison.regression_count} case(s) regressed**")

    if comparison.new_failures:
        st.warning("**New failure causes (appeared in current):**")
        for cause, count in comparison.new_failures.items():
            st.write(f"  - {cause}: {count}")

    if comparison.fixed_failures:
        st.success("**Fixed failure causes (no longer appear):**")
        for cause, count in comparison.fixed_failures.items():
            st.write(f"  - {cause}: {count} (was {count} in baseline)")

    # Case-by-case comparison table
    st.subheader("Case-by-Case Changes")
    table_data = []
    for diff in comparison.case_diffs:
        # Color-code by change type
        if diff.change == "improved":
            icon = "🟢"
        elif diff.change == "regressed":
            icon = "🔴"
        elif diff.change == "same":
            icon = "⚪"
        else:
            icon = "⚠️"

        table_data.append(
            {
                "Case": diff.case_index,
                "Change": f"{icon} {diff.change}",
                "Baseline": "✅" if diff.baseline_passed else "❌",
                "Current": "✅" if diff.current_passed else "❌",
                "Baseline Cause": diff.baseline_root_cause,
                "Current Cause": diff.current_root_cause,
            }
        )

    st.dataframe(table_data, use_container_width=True)

    # Detailed case inspector
    st.subheader("Drill-Down: Inspect a Case Diff")
    case_idx = st.selectbox(
        "Select a case to inspect:",
        options=range(len(comparison.case_diffs)),
        format_func=lambda i: f"Case {i}: {comparison.case_diffs[i].change}",
    )

    if case_idx is not None:
        diff = comparison.case_diffs[case_idx]
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Baseline")
            st.write(f"**Passed:** {'✅' if diff.baseline_passed else '❌'}")
            if diff.baseline_score is not None:
                st.write(f"**Score:** {diff.baseline_score:.3f}")
            st.write(f"**Root Cause:** `{diff.baseline_root_cause}`")

        with col2:
            st.subheader("Current")
            st.write(f"**Passed:** {'✅' if diff.current_passed else '❌'}")
            if diff.current_score is not None:
                st.write(f"**Score:** {diff.current_score:.3f}")
            st.write(f"**Root Cause:** `{diff.current_root_cause}`")

        # Change summary
        st.write(f"**Change Type:** {diff.change}")


def display_history(history: RunHistory):
    """Display a run history with trends and regression detection (Phase 4 mode)."""
    st.header("📈 History & Trends")

    if not history.entries:
        st.info("No runs in history.")
        return

    # List all runs in history
    st.subheader("All Runs")
    run_data = []
    for entry in history.entries:
        run_data.append(
            {
                "Run ID": entry.run_id,
                "Timestamp": entry.metadata.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Agent Version": entry.metadata.agent_version,
                "Pass Rate": f"{entry.result_set.metrics.pass_rate:.1%}",
                "Passed": entry.result_set.metrics.passed,
                "Failed": entry.result_set.metrics.failed,
                "Notes": entry.metadata.notes,
            }
        )
    st.dataframe(run_data, use_container_width=True)

    # Trend analysis
    st.subheader("Pass Rate Trend")
    try:
        trend = history.get_trend("pass_rate")
        if trend:
            # Convert to list of dicts for Streamlit
            trend_data = {"Run": [r[0] for r in trend], "Pass Rate": [r[1] for r in trend]}
            st.line_chart(data={"Pass Rate": [r[1] for r in trend]}, x_label="Run", y_label="Pass Rate (%)")

            # Display as table
            st.write("**Trend Data:**")
            trend_table = []
            for run_id, pass_rate in trend:
                trend_table.append({"Run": run_id, "Pass Rate": f"{pass_rate:.1%}"})
            st.dataframe(trend_table, use_container_width=True)
    except Exception as e:
        st.error(f"Error computing trend: {e}")

    # Regression detection
    st.subheader("Regression Detection")
    threshold = st.slider("Regression threshold (%)", min_value=1, max_value=20, value=5) / 100.0
    regressions = history.detect_regression(threshold=threshold)

    if regressions:
        st.error(f"**Regressions detected:** {', '.join(regressions)}")
        for run_id in regressions:
            entry = history.get_entry_by_run_id(run_id)
            if entry:
                st.write(f"  - {run_id}: {entry.metadata.notes}")
    else:
        st.success("**No regressions detected**")

    # Run selector for detailed view
    st.subheader("Inspect a Run")
    run_id = st.selectbox(
        "Select a run:", options=[e.run_id for e in history.entries], format_func=lambda x: f"{x}"
    )

    if run_id:
        entry = history.get_entry_by_run_id(run_id)
        if entry:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Timestamp:** {entry.metadata.timestamp}")
            with col2:
                st.write(f"**Agent Version:** {entry.metadata.agent_version}")
            with col3:
                st.write(f"**Judge Used:** {entry.metadata.judge_used}")

            st.write(f"**Notes:** {entry.metadata.notes}")

            # Show this run's detailed metrics
            result_set = entry.result_set
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pass Rate", f"{result_set.metrics.pass_rate:.1%}")
            with col2:
                st.metric("Passed", result_set.metrics.passed)
            with col3:
                st.metric("Failed", result_set.metrics.failed)
            with col4:
                st.metric("Errors", result_set.metrics.errors)


def main():
    st.set_page_config(page_title="Autopsy Evaluation Dashboard", layout="wide")
    st.title("📊 Autopsy Evaluation Dashboard")

    # Parse CLI args to determine mode
    mode = "single"  # default
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]

    # Mode selector in sidebar
    st.sidebar.header("🎯 Mode")
    available_modes = ["Single Run", "Compare Two", "View History"]
    mode_names = {"single": "Single Run", "compare": "Compare Two", "history": "View History"}
    selected_mode_display = st.sidebar.radio("Select mode:", available_modes)
    mode_map = {"Single Run": "single", "Compare Two": "compare", "View History": "history"}
    mode = mode_map.get(selected_mode_display, "single")

    # SINGLE RUN MODE
    if mode == "single":
        st.sidebar.header("📁 Data")
        data_file = None
        if "--data" in sys.argv:
            idx = sys.argv.index("--data")
            if idx + 1 < len(sys.argv):
                data_file = sys.argv[idx + 1]

        if data_file is None:
            data_file = st.sidebar.text_input("Path to BatchResultSet JSON:", value="")

        result_set = None
        if data_file:
            try:
                result_set = BatchResultSet.from_file(data_file)
                st.sidebar.success(f"✓ Loaded {result_set.total_cases} cases from {Path(data_file).name}")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")

        if result_set is None:
            st.info("Load a batch results JSON file to get started. See `examples/batch_run.py` for how to create one.")
            return

        display_single_run(result_set)

    # COMPARISON MODE
    elif mode == "compare":
        st.sidebar.header("📁 Data")

        baseline_file = None
        current_file = None

        if "--baseline" in sys.argv:
            idx = sys.argv.index("--baseline")
            if idx + 1 < len(sys.argv):
                baseline_file = sys.argv[idx + 1]

        if "--current" in sys.argv:
            idx = sys.argv.index("--current")
            if idx + 1 < len(sys.argv):
                current_file = sys.argv[idx + 1]

        if baseline_file is None:
            baseline_file = st.sidebar.text_input("Path to baseline JSON:", value="")
        if current_file is None:
            current_file = st.sidebar.text_input("Path to current JSON:", value="")

        baseline_set = None
        current_set = None

        if baseline_file:
            try:
                baseline_set = BatchResultSet.from_file(baseline_file)
                st.sidebar.success(f"✓ Baseline: {baseline_set.total_cases} cases")
            except Exception as e:
                st.sidebar.error(f"Error loading baseline: {e}")

        if current_file:
            try:
                current_set = BatchResultSet.from_file(current_file)
                st.sidebar.success(f"✓ Current: {current_set.total_cases} cases")
            except Exception as e:
                st.sidebar.error(f"Error loading current: {e}")

        if baseline_set is None or current_set is None:
            st.info("Load both baseline and current result sets to compare them.")
            return

        display_comparison(baseline_set, current_set)

    # HISTORY MODE
    elif mode == "history":
        st.sidebar.header("📁 Data")

        history_file = None
        if "--history" in sys.argv:
            idx = sys.argv.index("--history")
            if idx + 1 < len(sys.argv):
                history_file = sys.argv[idx + 1]

        if history_file is None:
            history_file = st.sidebar.text_input("Path to RunHistory JSON:", value="")

        history = None
        if history_file:
            try:
                history = RunHistory.from_file(history_file)
                st.sidebar.success(f"✓ Loaded history with {len(history.entries)} runs")
            except Exception as e:
                st.sidebar.error(f"Error loading history: {e}")

        if history is None:
            st.info("Load a RunHistory JSON file to view trends and detect regressions.")
            return

        display_history(history)


if __name__ == "__main__":
    main()

"""Streamlit dashboard for Phase 3 — visualize batch evaluation results.

Load a saved BatchResultSet JSON and explore:
  - Overall pass/fail rate
  - Failure distribution (pie chart)
  - Per-case drill-down (click a row to see full trace + diagnosis)

Usage:
    streamlit run src/autopsy/dashboard/app.py -- --data results.json
or
    streamlit run src/autopsy/dashboard/app.py -- --data results.json --dataset-dir .

Run without --data to show an empty dashboard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

try:
    from ..eval.aggregation import BatchResultSet
except ImportError:
    # When run as a Streamlit app, imports may need adjustment
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.autopsy.eval.aggregation import BatchResultSet


def main():
    st.set_page_config(page_title="Autopsy Evaluation Dashboard", layout="wide")
    st.title("📊 Autopsy Evaluation Dashboard")

    # Parse CLI args
    data_file = None
    if "--data" in sys.argv:
        idx = sys.argv.index("--data")
        if idx + 1 < len(sys.argv):
            data_file = sys.argv[idx + 1]

    # Sidebar: load data
    st.sidebar.header("📁 Data")
    if data_file is None:
        data_file = st.sidebar.text_input("Path to BatchResultSet JSON:", value="")

    result_set = None
    if data_file:
        try:
            result_set = BatchResultSet.from_file(data_file)
            st.sidebar.success(f"✓ Loaded {result_set.total_cases} cases from {Path(data_file).name}")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            result_set = None

    # If no data, show empty state
    if result_set is None:
        st.info(
            "Load a batch results JSON file to get started. "
            "See `examples/batch_run.py` for how to create one."
        )
        return

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

    # Pass/fail pie
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
            table_data.append({
                "Index": case.case_index,
                "Status": status,
                "Score": f"{case.overall_score:.2f}" if case.overall_score else "—",
                "Root Cause": case.root_cause,
                "Confidence": f"{case.confidence:.0%}" if case.confidence else "—",
                "Failures": ", ".join(case.failure_labels) if case.failure_labels else "—",
            })

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
                st.write(f"**Confidence:** {case.confidence:.0%}" if case.confidence else "**Confidence:** —")
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
                    metric_data.append({
                        "Metric": result.get("name", "—"),
                        "Score": f"{result.get('score', 0):.2f}" if result.get("score") is not None else "—",
                        "Passed": "✓" if result.get("passed") else "✗" if result.get("passed") is False else "—",
                        "Failure Label": result.get("failure_label", "—"),
                        "Rationale": result.get("rationale", "—")[:60] + "…",
                    })
                st.dataframe(metric_data, use_container_width=True)


if __name__ == "__main__":
    main()

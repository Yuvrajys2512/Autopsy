# Phase 4: Result Comparison, History & Advanced Dashboard

**Current state**: Phase 3 complete (61 tests passing, datasets + batch runner + basic Streamlit dashboard working). All code in `src/autopsy/eval/` and `src/autopsy/dashboard/`. Main branch clean.

## Phase 4 Gate

> Compare batch results across time or variants (agent versions, prompt changes, dataset updates). Track trends. Suggest improvements based on failure patterns. Advanced dashboard with experiment selection, result diffing, and failure drill-down with traces.

### Success criteria

- [ ] `comparison.py` module: `compare(baseline_json, current_json)` → diff metrics, regression detection
- [ ] `history.py` module: Track batch runs over time, compute trends (pass rate improvement, failure shifts)
- [ ] Enhanced dashboard: Load multiple BatchResultSet JSONs, select baseline vs. current, show diffs
- [ ] 15+ new tests (should reach 76+ total)
- [ ] `examples/compare_runs.py` demonstrating baseline vs. experiment comparison
- [ ] `examples/track_trends.py` showing history analysis (3+ runs, trend detection)
- [ ] Documentation: `documentation/phase-4-comparison/` (6 files, same pattern as Phase 3)
- [ ] No breaking changes to Phase 1-3 APIs

## What to Build

### 1. Comparison Module (`src/autopsy/eval/comparison.py`)

```python
class ComparisonMetrics(BaseModel):
    """Metrics comparing baseline vs. current."""
    baseline_pass_rate: float
    current_pass_rate: float
    pass_rate_delta: float  # current - baseline (positive = improvement)
    regression: bool  # True if any case went from pass to fail
    improved: bool  # True if any case went from fail to pass
    new_failures: dict[str, int]  # failure causes that appeared in current but not baseline
    fixed_failures: dict[str, int]  # failure causes that disappeared
    case_diffs: list[CaseDiff]  # per-case changes

class CaseDiff(BaseModel):
    """One case's change between baseline and current."""
    case_index: int
    trace_id_baseline: str
    trace_id_current: str
    baseline_passed: bool
    current_passed: bool
    baseline_score: Optional[float]
    current_score: Optional[float]
    baseline_root_cause: str
    current_root_cause: str
    change: str  # "improved" | "regressed" | "same" | "baseline_error" | "current_error"

def compare(baseline: BatchResultSet, current: BatchResultSet) -> ComparisonMetrics:
    """Compare two batch result sets.
    
    Args:
        baseline: Previous run (baseline).
        current: New run (experiment).
    
    Returns:
        ComparisonMetrics with diffs and summary.
    
    Raises:
        ValueError: If datasets don't match (different case counts or order).
    """
```

### 2. History Module (`src/autopsy/eval/history.py`)

```python
class BatchRunMetadata(BaseModel):
    """Metadata for one batch run."""
    run_id: str  # e.g., "2026-01-15T10:30:00Z" or "exp-v2.1-gpt4"
    timestamp: datetime
    dataset_name: str
    dataset_version: str
    agent_version: str  # e.g., "v1.0", "main", "exp-branch"
    judge_used: bool
    notes: str  # e.g., "new prompt", "temperature=0.7"

class HistoryEntry(BaseModel):
    """One run in a history."""
    run_id: str
    metadata: BatchRunMetadata
    result_set: BatchResultSet

class RunHistory(BaseModel):
    """A sequence of runs over time."""
    name: str  # e.g., "agent-experiments"
    entries: list[HistoryEntry] = Field(default_factory=list)
    
    def add_run(self, run_id: str, metadata: BatchRunMetadata, result_set: BatchResultSet) -> None:
        """Add a run to the history."""
        self.entries.append(HistoryEntry(run_id=run_id, metadata=metadata, result_set=result_set))
    
    def get_trend(self, metric: str) -> list[tuple[str, float]]:
        """Get a time-series of a metric (e.g., 'pass_rate').
        
        Returns:
            List of (run_id, value) tuples in chronological order.
        """
    
    def detect_regression(self, threshold: float = 0.05) -> list[str]:
        """Detect runs where pass_rate dropped > threshold from previous.
        
        Args:
            threshold: Max allowed drop (e.g., 0.05 = 5%).
        
        Returns:
            List of run_ids with regressions.
        """
    
    def save(self, path: str | Path) -> None:
        """Save history to JSON."""
    
    @classmethod
    def from_file(cls, path: str | Path) -> RunHistory:
        """Load history from JSON."""
```

### 3. Advanced Dashboard (`src/autopsy/dashboard/app.py` — enhance existing)

**New UI flows**:

#### A. Single-run mode (existing, keep as-is)
- Load one BatchResultSet
- Show overview, charts, drill-down

#### B. Comparison mode (new)
- Load two BatchResultSet JSONs (baseline and current)
- Show side-by-side metrics
- Highlight regressions (red), improvements (green), unchanged (gray)
- Case-by-case diff table with per-case change indicator
- Drill-down to see what changed in a regressed case

#### C. History mode (new)
- Load a RunHistory JSON (or multiple BatchResultSet JSONs with metadata)
- Show pass-rate trend line
- Show failure-distribution heatmap over time (rows = failure causes, columns = runs)
- Regression detection: highlight runs with drops
- Select a run to inspect in detail

**Technical**:
- Radio buttons at top: "Single Run | Compare Two | View History"
- Conditional rendering based on selection
- Charts: line plot for trends (use `st.line_chart()` or Plotly if available)

### 4. Tests (`tests/test_phase4.py`)

```python
# Comparison tests (6+)
def test_compare_identical_runs():
    """Comparing identical runs should show zero delta."""

def test_compare_detects_improvement():
    """If current has higher pass_rate, delta is positive."""

def test_compare_detects_regression():
    """If any case went pass→fail, regression=True."""

def test_compare_case_diff_logic():
    """CaseDiff correctly categorizes improvements vs regressions."""

def test_compare_failure_cause_shifts():
    """new_failures and fixed_failures correctly track cause changes."""

def test_compare_mismatched_case_counts():
    """Comparing datasets with different case counts raises ValueError."""

# History tests (6+)
def test_run_history_add_run():
    """Adding runs to history maintains order."""

def test_run_history_get_trend():
    """Trend extraction gives correct time-series."""

def test_run_history_detect_regression():
    """Regression detection flags runs below threshold."""

def test_run_history_save_and_load():
    """History round-trips to JSON without loss."""

def test_run_history_filter_by_metadata():
    """Can query history by agent_version or notes."""

def test_run_history_empty():
    """Empty history doesn't crash."""

# Aggregation tests (3+)
def test_dashboard_compare_mode_loads():
    """Dashboard can load two BatchResultSets and show comparison."""

def test_dashboard_history_mode_loads():
    """Dashboard can load RunHistory and display trends."""

def test_dashboard_regression_highlight():
    """Regression rows are visually marked (red, etc)."""
```

### 5. Examples

#### `examples/compare_runs.py`
- Run two different agents (or two versions of the same agent) on the same dataset
- Generate two BatchResultSet JSONs
- Use `compare()` to get ComparisonMetrics
- Print a summary: "Pass rate improved 5% (70% → 75%)", "2 regressions", "3 improvements"
- Save comparison to JSON

#### `examples/track_trends.py`
- Create a RunHistory
- Simulate 3-5 runs with different versions (v1.0, v1.1, v2.0, etc.)
- Add each run's BatchResultSet to history
- Call `get_trend("pass_rate")` and print a simple ASCII line chart
- Detect and print any regressions
- Save history to JSON

### 6. Documentation (`documentation/phase-4-comparison/` — 6 files)

Follow Phase 3 structure:
1. **01-what-we-built-and-why.md** — Why result comparison matters, experiment tracking, continuous improvement
2. **02-theory-a-b-testing-and-trends.md** — Statistical rigor, trend detection, regression thresholds, confidence intervals
3. **03-tech-stack.md** — Comparison engine, history storage, dashboard charting library choice
4. **04-architecture-and-system-design.md** — Comparison algorithm, history time-series, dashboard data flow
5. **05-design-decisions.md** — Sequential diff vs. summarization, threshold choices, history versioning
6. **06-code-walkthrough.md** — Detailed code inspection of comparison.py, history.py, dashboard enhancements

## Hard Constraints

- **No breaking changes** to Phase 1-3 APIs. `batch_evaluate()`, `evaluate()`, `classify()` unchanged.
- **No new dependencies** for `comparison.py` and `history.py` (Pydantic only). Dashboard can use `plotly` or `altair` if needed (optional, like Streamlit).
- **Comparison is deterministic** — same two BatchResultSets always produce the same ComparisonMetrics.
- **History is append-only** — runs can't be deleted or modified (immutability aids debugging).
- **Tests**: 15+ new tests, all Phase 4-specific. Should reach 76+ total.
- **No commits** — suggest title when done.

## Edge Cases to Handle

- **Comparing runs on different datasets** — should fail gracefully with clear error.
- **Comparing runs with different case counts** — error.
- **Case reordering** — if baseline and current have same cases but different order, compare by trace_id or case index?
- **History with gaps** — missing runs in a time-series (e.g., runs 1, 2, 4 — where's 3?). Should still work.
- **Regression detection with noisy metrics** — a 1% drop might be noise; allow threshold tuning.
- **Empty history** — handle gracefully (show placeholder, don't crash).

## Learning Outcomes

- **Experiment design** — how to structure A/B tests for agents.
- **Trend detection** — identifying regressions and improvements at scale.
- **Comparative analysis** — framing results relative to a baseline.
- **Time-series analysis** — tracking metrics over many runs.
- **UX for data exploration** — making complex comparisons intuitive.

## Example Usage (what Phase 4 enables)

```python
from autopsy import batch_evaluate, Dataset
from autopsy.eval.comparison import compare
from autopsy.eval.history import RunHistory, BatchRunMetadata
from autopsy.eval.aggregation import BatchResultSet
from datetime import datetime

# Run baseline
dataset = Dataset.from_file("benchmark.json")
baseline_results = batch_evaluate(agent_v1, dataset)
baseline_set = BatchResultSet.from_batch_results("benchmark", "1.0.0", baseline_results)
baseline_set.save("baseline.json")

# Run experiment (new prompt)
exp_results = batch_evaluate(agent_v2, dataset)
exp_set = BatchResultSet.from_batch_results("benchmark", "1.0.0", exp_results)
exp_set.save("exp_new_prompt.json")

# Compare
comparison = compare(baseline_set, exp_set)
print(f"Pass rate: {comparison.baseline_pass_rate:.0%} → {comparison.current_pass_rate:.0%}")
print(f"Delta: {comparison.pass_rate_delta:+.1%}")
if comparison.regression:
    print(f"WARNING: {len([d for d in comparison.case_diffs if d.change == 'regressed'])} regressions")

# Track history
history = RunHistory(name="agent-experiments")
history.add_run(
    run_id="baseline",
    metadata=BatchRunMetadata(
        run_id="baseline",
        timestamp=datetime.now(),
        dataset_name="benchmark",
        dataset_version="1.0.0",
        agent_version="v1.0",
        judge_used=False,
        notes="Original prompt",
    ),
    result_set=baseline_set,
)
history.add_run(
    run_id="exp-new-prompt",
    metadata=BatchRunMetadata(
        run_id="exp-new-prompt",
        timestamp=datetime.now(),
        dataset_name="benchmark",
        dataset_version="1.0.0",
        agent_version="v2.0",
        judge_used=False,
        notes="Improved prompt with examples",
    ),
    result_set=exp_set,
)
history.save("experiment_history.json")

# View in dashboard
# streamlit run src/autopsy/dashboard/app.py -- --mode compare --baseline baseline.json --current exp_new_prompt.json
# OR
# streamlit run src/autopsy/dashboard/app.py -- --mode history --history experiment_history.json
```

## Key Files to Know

- `src/autopsy/eval/dataset.py` — TestCase, Dataset (Phase 3)
- `src/autopsy/eval/batch.py` — batch_evaluate, BatchResult (Phase 3)
- `src/autopsy/eval/aggregation.py` — BatchResultSet, BatchCaseResult, AggregateMetrics (Phase 3)
- `src/autopsy/eval/comparison.py` — **NEW**: ComparisonMetrics, compare()
- `src/autopsy/eval/history.py` — **NEW**: RunHistory, BatchRunMetadata, HistoryEntry
- `src/autopsy/dashboard/app.py` — Enhance with compare & history modes
- `tests/test_phase3.py` — Existing (don't break)
- `tests/test_phase4.py` — **NEW**: All Phase 4 tests

## Definition of Done

- [ ] `comparison.py` with `compare()` function and `ComparisonMetrics` model
- [ ] `history.py` with `RunHistory` and `BatchRunMetadata` models
- [ ] Dashboard enhanced: comparison mode + history mode working
- [ ] 15+ Phase 4 tests passing (76+ total)
- [ ] `examples/compare_runs.py` demonstrates baseline vs. experiment comparison
- [ ] `examples/track_trends.py` demonstrates history and trend detection
- [ ] `documentation/phase-4-comparison/` (6 files)
- [ ] All Phase 1-3 tests still pass (no regressions)

## Notes for Claude

- Reuse Phase 1-3 patterns: Pydantic models, graceful error handling, pluggable abstractions
- Comparison should be referential — compare two BatchResultSet objects in-memory, don't modify them
- History can store full BatchResultSet or just metadata + reference to saved JSON (your call)
- Dashboard: start simple (line chart for trends, side-by-side metrics for comparison); fancy features are nice-to-have
- No commits; suggest title when done

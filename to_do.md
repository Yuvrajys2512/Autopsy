# Autopsy — TODO (next session)

Picks up Phase 5 (Dogfood & Launch). The library is feature-complete (Phases 0–4,
**87 tests passing**) and packaged. What's left is running Autopsy on a real
third-party agent and publishing the finding. Full strategy lives in `PHASE5_PLAN.md`.

---

## Already done this session (no action needed)

- **API:** Phase 4 modules (`comparison`, `history`, `aggregation`) exported from `autopsy` / `autopsy.eval`.
- **README:** status table corrected; classify / batch / dashboard / compare / history all documented.
- **Packaging:** `pyproject.toml` fixed (correct repo URLs, classifiers, `dashboard` extra), added `LICENSE` + `py.typed`. Built sdist+wheel, `twine check` PASSED, clean-room `pip install` verified. `dist/`+`build/` gitignored.
- **Phase 5 groundwork:**
  - `examples/build_dogfood_dataset.py` → `examples/datasets/multihop_qa.json` (24 multi-hop QA cases, smolagents tools, budget=8).
  - `examples/dogfood_smolagents.py` — real smolagents instrumentation + dependency-free mock fallback.
  - Validated end-to-end via mock: 67% pass, 6 distinct failure modes.
  - **Bug fixed:** `from_batch_results` was counting passing cases in `failure_distribution` (bogus `goal_not_completed` inflation) — gated on non-passing cases + regression test added.

---

## TODO

### 1. Publish to PyPI (irreversible per version — do when ready)
Build is already validated. Recommend TestPyPI first:
```bash
.venv/Scripts/python.exe -m build                      # already built; rebuild if changed
.venv/Scripts/python.exe -m twine upload --repository testpypi dist/*
# then the real index:
.venv/Scripts/python.exe -m twine upload dist/*
```
Needs a PyPI account + API token. After upload, confirm `pip install autopsy` in a fresh venv.

### 2. Run the REAL smolagents dogfood (the headline)
Needs smolagents + an Anthropic key on your machine:
```bash
pip install "smolagents[toolkit]" anthropic
$env:ANTHROPIC_API_KEY="sk-ant-..."      # PowerShell  (bash: export ANTHROPIC_API_KEY=...)
python examples/dogfood_smolagents.py     # auto-detects real mode (else falls back to mock)
```
- Output: `examples/datasets/runs/smolagents_run.json` (a `BatchResultSet`).
- **Caveat:** the instrumentation targets smolagents' current `ToolCallingAgent` / `WebSearchTool` / `VisitWebpageTool` / `LiteLLMModel` API. If your installed version differs, the tool/model wrapping in `dogfood_smolagents.py` (`_instrument_tool`, `_instrument_model`, `build_real_agent_fn`) may need a small tweak — paste Claude the error next session.
- Sanity-check: spot-check a few gold answers in the dataset still hold.

### 3. Capture the dashboard screenshot/GIF (README gate)
```bash
streamlit run src/autopsy/dashboard/app.py -- --mode single --data examples/datasets/runs/smolagents_run.json
```
Screenshot the failure breakdown + a drill-down; add the GIF to the top of the README.

### 4. (Optional) Second story — comparison run
Run the dataset twice (e.g. two prompts or two models), `compare()` the two `BatchResultSet`s, and report the regression/improvement diff. Would need a new `examples/compare_dogfood.py`. Ask Claude to write it.

### 5. Write + publish the launch post
Lead with the finding, not the features. Template:
> "I ran Autopsy on a smolagents agent over 24 HotpotQA-style tasks and found N% of its
> failures were \<X\> that a plain pass/fail score never surfaces."
Structure: finding → failure-distribution chart → 1–2 annotated trace drill-downs (culprit span) →
`pip install autopsy` + 10-line quickstart → repo link. Post to LinkedIn first, then Show HN / the smolagents community.

---

## Gate checklist (from the brief)
- [ ] `pip install autopsy` works for a stranger  *(built & verified locally; needs the PyPI upload)*
- [ ] Stranger → eval report in <15 min from the README alone
- [ ] README leads with a dashboard failure-breakdown GIF
- [ ] Run on ≥1 real agent that isn't ours, with published numbers
- [ ] Launch post live with a concrete finding
- [ ] **True signal:** someone else stars it / opens an issue, unprompted

---

## Uncommitted work to stage first
Suggested commit: `Phase 5: dogfood dataset, smolagents harness, fix failure-distribution bug`
Touches: `src/autopsy/__init__.py`, `src/autopsy/eval/__init__.py`, `src/autopsy/eval/aggregation.py`,
`README.md`, `pyproject.toml`, `LICENSE`, `src/autopsy/py.typed`, `.gitignore`,
`examples/build_dogfood_dataset.py`, `examples/dogfood_smolagents.py`,
`examples/datasets/multihop_qa.json`, `tests/test_phase3.py`, `PHASE5_PLAN.md`, `to_do.md`.

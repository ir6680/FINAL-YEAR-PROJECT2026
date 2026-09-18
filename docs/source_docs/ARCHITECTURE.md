# Architecture

The project uses a layered structure:

- Entry points: `app.py` for Streamlit and `main.py` for CLI execution.
- Configuration: `src/config/settings.py` defines canonical paths.
- ML layer: `src/ml/` handles clustering, classification, drift detection, and explainability.
- Finance layer: `src/finance/` handles derived metrics, policy guardrails, recommendation, and portfolio allocation.
- Audit layer: `src/blockchain/` creates simulated blockchain receipts.
- Simulation layer: `src/simulation/` evaluates simple 12-month portfolio outcomes.

The large Streamlit app is intentionally preserved to avoid changing UI/session-state behavior during this structure-only refactor.

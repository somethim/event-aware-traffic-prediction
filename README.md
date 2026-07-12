# Event-Aware Traffic Prediction

BSc thesis project. **Research question:** can a machine-learning traffic
prediction model be made more accurate by feeding it information about planned
public events (concerts, sports matches, large gatherings) in addition to
historical traffic data?

The project trains and compares two configurations of the *same* model type so
that the **only** variable is the event information:

| Run | Model | Features | Purpose |
|-----|-------|----------|---------|
| A  | Traffic model (baseline)    | temporal + historical/lag flow | control |
| A+ | Traffic model (event-aware) | A's features **+ event-impact score** | treatment |

The event-impact score itself is produced by a second model:

| Model | Input | Output |
|-------|-------|--------|
| B (event impact) | event metadata relative to a sensor & time (distance, expected popularity, time-to-event, …) | a scalar "event pressure" score |

So the flow is: **B → feeds A+**, and we compare **A vs A+** on an identical
test set. This directly tests the thesis hypothesis: including event-based data
(location, time, expected popularity) improves accuracy, *especially* during
event-driven congestion.

## Model choice

- **Random Forest** (scikit-learn) is the default for both the traffic model and
  the event-impact model. It is a strong, low-tuning baseline for tabular data
  and — importantly for the thesis — exposes **feature importances**, so you can
  quantify how much the event feature actually contributes.
- Swappable to **Gradient Boosting** via `config/config.yaml` (`model.type`).
- Both A and A+ **must** use the same `model.type` — the comparison is only valid
  if the model is held constant and the feature set is the only difference.

## Data

The pipeline runs end-to-end **today** on a synthetic PeMS-style dataset that the
generator creates for you (traffic *flow* per sensor per timestamp, plus a table
of planned events). The synthetic generator injects a known, ground-truth event
effect into the flow, so there is a real signal for Model B to learn and for the
comparison to reveal.

To move to **real data** later (thesis "real-world data" requirement):
- Traffic: swap in **PEMS-BAY / PeMS** flow data (see `src/data/load.py`).
- Events: implement the scraper in `src/data/scrape_events.py` (Ticketmaster
  Discovery API, or scrape a local events/venue calendar for your metro).

## Quickstart

Uses [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned (`.python-version`);
uv fetches it automatically.

```bash
# 1. install deps into a managed venv
uv sync --extra dev

# 2. run the whole thing (generate data -> train A, B, A+ -> compare)
uv run python -m scripts.run_all
# ...or step by step:
uv run python -m src.data.generate_synthetic     # writes data/synthetic/*.parquet
uv run python -m src.pipeline.prepare            # cached feature matrix
uv run python -m src.pipeline.train_baseline     # Model A
uv run python -m src.pipeline.train_event_model  # Model B  (writes event_impact_score)
uv run python -m src.pipeline.train_event_aware  # Model A+
uv run python -m src.pipeline.compare            # metrics + plots -> results/

# benchmark EVERY model (RF, GB, XGBoost, ...) — each run as A and A+ — and compare
uv run python -m scripts.run_benchmark           # -> results/benchmark.json + benchmark.png

# tests
uv run pytest
```

### Models & GPU

`model.type` in `config/config.yaml` selects the estimator; **both A and A+ always use
the same type** so the event feature is the only variable.

| `model.type` | Backend | GPU |
|--------------|---------|-----|
| `random_forest` (default) | scikit-learn | CPU only |
| `gradient_boosting` | scikit-learn | CPU only |
| `xgboost` | XGBoost | **GPU** via `model.device` |

For XGBoost, `model.device` is `auto` (use a CUDA GPU if one is detected, else CPU),
or force `cuda` / `cpu`. Detection lives in `src/device.py`. Note: at this dataset size
(~10⁵ rows) GPU may not beat CPU — the benchmark reports inference latency so you can
show the CPU-vs-GPU trade-off, which feeds the thesis's *response time* criterion.

The **benchmark** (`scripts.run_benchmark`) trains every model listed under
`benchmark.models` in the config as both A and A+, then prints one comparison table and
writes `results/benchmark.json` + `results/benchmark.png`. Add a model type to that list
and it joins the comparison automatically. Data prep and Model B run once (they don't
depend on the traffic model), so only runs A/A+ repeat per model.

Optional extras: `uv sync --extra scrape` (event scraper deps),
`uv sync --extra boosting` (xgboost/lightgbm).

### Development tooling

```bash
uv run black src scripts tests     # format (line length 100)
uv run mypy                        # static type check (src, scripts, tests)
uv run pytest                      # smoke tests
```

Config lives in `pyproject.toml` (`[tool.black]`, `[tool.mypy]`). Note: `pytest`
regenerates a small synthetic dataset into `data/`, so run `scripts.run_all`
afterwards to restore the full dataset (all data is synthetic and gitignored).

Outputs land in `results/` (`metrics.json`, comparison plots) and trained models
in `models/`.

## Layout

```
config/config.yaml          all knobs (data size, model type, features)
src/data/                   synthetic generation, real-data loaders, event scraper (stub)
src/features/               feature engineering (temporal, lags, event exposure)
src/models/                 traffic model, event-impact model, metrics
src/pipeline/               train A / B / A+  and  compare
scripts/run_all.py          one-shot runner
results/                    metrics + figures for the thesis
```

## Thesis evaluation criteria (from the abstract)

- **Accuracy** — MAE, RMSE, MAPE, R² (overall and *on event-affected rows*).
- **Reliability** — error distribution / worst-case error, especially at event peaks.
- **Response time** — model inference latency (recorded in `metrics.json`).

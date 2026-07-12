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

`data.source` selects the dataset (both write the same parquet schema, so the pipeline is
source-agnostic):

- **`real`** (default) — **Caltrans PeMS** loop-detector flow + **setlist.fm** historical
  concerts, i.e. the thesis's "real-world data".
  - **PeMS** (`src/data/pems.py`): the raw 5-min data (~2.7 GB) is published on Google Drive and
    **auto-downloaded** (via `gdown`) into `data/raw/pems/` when missing — it is not kept in the
    repo. The loader reads flow/speed/occupancy + `%Observed`, drops imputed bins, and
    `data.target` picks which of flow/speed/occupancy the model predicts.
  - **setlist.fm** (`src/data/setlistfm.py`): a free, *historical* concert API (so it can cover a
    past PeMS window). It gives date-only + city-level coords + no capacity, so a **venue
    reference table** supplies each major LA/SF venue's precise `(lat, lon, capacity)` and a
    default local start hour (20:00). Venue **capacity** is the popularity signal; concerts at
    unlisted venues are dropped. Start times are on PeMS's Pacific-local clock, so events line up
    with the flow timestamps.
  - Build/refresh it explicitly with `uv run python -m scripts.build_real_dataset` (needs
    `SETLISTFM_API_KEY` in `.env`); the runners also build it once if missing.
- **`synthetic`** (opt-in) — the generator in `src/data/generate_synthetic.py` fabricates
  PeMS-style flow with a **known, ground-truth event effect** injected. Useful as a *controlled
  testbed* (a model blind to events cannot explain the injected spikes) and for the offline
  tests. Set `data.source: synthetic` to use it.

> **Live deployment (out of scope, future work).** A real-time system would pair an upcoming-event
> feed (e.g. Ticketmaster Discovery API) with a *live* traffic feed. There is no free live
> equivalent of PeMS per-sensor flow: Google Maps only exposes travel time in traffic (Routes API,
> paid), and TomTom/HERE offer real-time segment speed on freemium tiers. This project is
> therefore an **offline historical study**; live inference is left as future work.

## Quickstart

Uses [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned (`.python-version`);
uv fetches it automatically.

```bash
# 1. install deps into a managed venv
uv sync --extra dev

# 2. build the dataset (real by default: auto-fetches PeMS from Drive + setlist.fm events)
uv run python -m scripts.build_real_dataset      # needs SETLISTFM_API_KEY in .env
#    (or use the controlled synthetic testbed instead: set data.source: synthetic in config)

# 3. run EVERYTHING: experiment matrix (all models × conditions) + headline pipeline + figures
uv run python -m scripts.run_all         # heavy (several min); fills docs/thesis-notes.md + media/
# ...or the individual pieces:
uv run python -m src.pipeline.prepare            # cached feature matrix
uv run python -m src.pipeline.train_baseline     # Model A
uv run python -m src.pipeline.train_event_model  # Model B  (writes event_impact_score)
uv run python -m src.pipeline.train_event_aware  # Model A+
uv run python -m src.pipeline.compare            # metrics + plots -> media/

# benchmark EVERY model (RF, GB, XGBoost, ...) — each run as A and A+ — and compare
uv run python -m scripts.run_benchmark           # -> media/results/benchmark.json + media/figures/

# statistical rigor: rolling-origin CV x seeds + significance test on the A->A+ gap
uv run python -m scripts.run_stats               # -> media/results/stats.json

# generate all thesis figures (matplotlib) into media/figures/
uv run python -m scripts.run_visuals

# run the full experiment matrix (split × normalize × model) -> table in docs/thesis-notes.md
uv run python -m scripts.run_experiments

# tests
uv run pytest
```

Outputs land under **`media/`**: metrics JSON in `media/results/`, PNG figures in
`media/figures/`. (`data/` holds only inputs + intermediate parquet; both are gitignored.)

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
writes `media/results/benchmark.json` + `media/figures/benchmark.png`. Add a model type to that list
and it joins the comparison automatically. Data prep and Model B run once (they don't
depend on the traffic model), so only runs A/A+ repeat per model.

Optional extra: `uv sync --extra boosting` (lightgbm; xgboost is already a core dependency).
The setlist.fm/PeMS fetch deps (`requests`, `gdown`) are core, so no extra is needed for real data.

### Development tooling

```bash
uv run black src scripts tests     # format (line length 100)
uv run mypy                        # static type check (src, scripts, tests)
uv run pytest                      # smoke tests
```

**Everything in one line** (format → type-check → test → run the full matrix + figures):

```bash
uv run black src scripts tests && uv run mypy && uv run pytest && uv run python -m scripts.run_all
```

Config lives in `pyproject.toml` (`[tool.black]`, `[tool.mypy]`). The tests run on a small
synthetic dataset written to an isolated temp directory (`tests/conftest.py` points
`EATP_DATA_ROOT` there), so running `pytest` never touches your real `data/` — you can run it
mid-experiment safely.

Outputs land in `media/` (`results/metrics.json`, figures in `figures/`) and trained models
in `models/`.

## Layout

```
config/config.yaml          all knobs (data size, model type, features)
src/data/                   PeMS + setlist.fm loaders, Drive fetch/build, synthetic generator
src/features/               feature engineering (temporal, lags, event exposure)
src/models/                 traffic model, event-impact model, metrics
src/pipeline/               train A / B / A+  and  compare
scripts/run_all.py          one-shot runner
media/results/             metrics JSON
media/figures/             generated figures
```

## Thesis evaluation criteria (from the abstract)

- **Accuracy** — MAE, RMSE, MAPE, R² (overall and *on event-affected rows*).
- **Reliability** — error distribution / worst-case error, especially at event peaks.
- **Response time** — model inference latency (recorded in `metrics.json`).

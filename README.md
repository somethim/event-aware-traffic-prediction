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
| A+raw | Traffic model (event-aware) | A's features **+ raw planned-event features** | primary treatment |

The learned event-impact score is retained only as a diagnostic ablation:

| Model | Input | Output |
|-------|-------|--------|
| B (event impact) | event metadata relative to a sensor & time (distance, expected popularity, time-to-event, …) | a scalar "event pressure" score |

The predeclared comparison is **A vs A+raw** on identical rows and forward test
blocks. This tests whether including event-based data
(location, time, expected popularity) improves accuracy, *especially* during
event-driven congestion.

## Model choice

- **Random Forest** (scikit-learn) is the default for both the traffic model and
  the event-impact model. It is a strong, low-tuning baseline for tabular data
  and — importantly for the thesis — exposes **feature importances**, so you can
  quantify how much the event feature actually contributes.
- **XGBoost** is the robustness model. Slow scikit-learn Gradient Boosting is excluded from the
  claim-bearing run.
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
    `data.target` picks which of flow/speed/occupancy the model predicts. Note that PeMS
    *pre-fills* dead detectors with plausible-looking historical averages (flagged only via
    `%Observed`), so the gate un-fills them back to missing rather than training on invented
    values that by construction contain no event spikes; the build prints per-district
    retention stats. Detector health is bimodal (alive ≈ complete, dead ≈ fully imputed), so
    the 50 % threshold is not sensitive.
  - **setlist.fm** (`src/data/setlistfm.py`): a free, *historical* concert API (so it can cover a
    past PeMS window). It gives date-only + city-level coords + no capacity, so a **venue
    reference table** supplies each major LA/SF venue's precise `(lat, lon, capacity)` and a
    default local start hour (20:00). Venue **capacity** is the popularity signal; concerts at
    unlisted venues are dropped. Start times are on PeMS's Pacific-local clock, so events line up
    with the flow timestamps. Two validation gates keep the table honest: results are checked
    against the setlist's *own* city (the venue-name search matches nationwide — The Fillmore
    exists in half a dozen US cities), and multi-artist bills are collapsed to one event per
    venue-day (setlist.fm returns one setlist per performing artist).
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

# 2. fast implementation/mechanism validation (21 days, 12 sensors, 24 events)
uv run python -m scripts.run_synthetic_smoke --config config/synthetic.yaml

# 3. claim-bearing real study; requires SETLISTFM_API_KEY and may run overnight
uv run python -m scripts.run_real_thesis --config config/real.yaml

# tests
uv run pytest
```

Outputs land under **`runs/<run_id>/`**. `runs/latest-verified` changes only after all acceptance
checks pass. Generated data, models, results, and figures are gitignored and rebuilt as needed.

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
and it joins the comparison automatically. Data prep and Model B run once, so only runs A/A+
repeat per model. (On real data Model B's proxy target is derived from a baseline of the *base*
config's `model.type`; all per-model A/A+ runs then share that one score, which keeps the
comparison consistent across models.)

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
docs/thesis-notes.md        methodology notes + auto-generated experiment table (paper source)
src/data/                   PeMS + setlist.fm loaders, Drive fetch/build, synthetic generator
src/features/               feature engineering (temporal, lags, event exposure)
src/models/                 traffic model, event-impact model, metrics
src/pipeline/               train A / B / A+, compare, benchmark, stats CV, figures
scripts/                    thin runners (run_all, run_benchmark, run_stats, run_visuals, ...)
tests/                      offline test suite (isolated tmp dir; never touches data/)
media/results/              metrics JSON (metrics, benchmark, experiments, stats)
media/figures/              generated figures
```

## Reproducing the thesis results (checklist)

1. `uv sync --extra dev` and put `SETLISTFM_API_KEY` in `.env` (only needed to rebuild events).
2. `uv run python -m scripts.build_real_dataset` — skip if `data/processed_inputs/*.parquet`
   already exist (the runners also build it on demand).
3. `nohup uv run python -m scripts.run_all > run_all.log 2>&1 &` — experiment matrix +
   headline pipeline + figures. Hours on real data; safe to leave unattended.
4. `uv run python -m scripts.run_stats` — significance test + CI for the A→A+ gap.
5. Collect: `media/results/*.json`, `media/figures/*.png`, and the auto-filled experiment
   table in `docs/thesis-notes.md` §4.2. For the synthetic-only mechanism figures, set
   `data.source: synthetic` in the config and rerun `run_all` once.

## Thesis evaluation criteria (from the abstract)

- **Accuracy** — MAE, RMSE, MAPE, R² (overall and *on event-affected rows*).
- **Reliability** — error distribution / worst-case error, especially at event peaks.
- **Response time** — model inference latency (recorded in `metrics.json`).

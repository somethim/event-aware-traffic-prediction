# Thesis reference notes — Event-Aware Traffic Prediction

Working notes to draw on when writing the thesis paper. Structured as: design decisions
(with rationale), how the models train, current results, limitations, future work, and
references. Reference markers like `[1]` point to the **References** section at the end.

> Citation caveat: bibliographic entries below give authors / title / venue / year so you
> can locate each source. **Verify exact page numbers, DOIs, and access dates before final
> submission** and reformat to your required citation style (IEEE / APA / etc.).

> **Result status (verified 2026-07-26):** the claim-bearing real-data run is
> **`real-20260725T210943Z`**. It completed all six predeclared cells (three targets × Random
> Forest/XGBoost), wrote `results/fold_results.json` and `run_manifest.json`, and is the target
> of `runs/latest-verified`. Its config checksum is
> `dac659f8f61eda678f0eaa12ea1d97c8480627bb1b5db273f7888a18c130b320`; the manifest records
> source commit `a262951976b8b6211790ac1348136ad379e2a5aa`. Legacy files under
> `media/results/` and the older July 13 matrix are not claim-bearing.

## Reproducibility commands for final tables

```bash
uv run python -m scripts.run_synthetic_smoke --config config/synthetic.yaml
uv run python -m scripts.run_sample_study --config config/sample.yaml
uv run python -m scripts.run_real_thesis --config config/real.yaml
uv run pytest -q
uv run mypy
uv run black --check src scripts tests
```

The real runner uses five non-overlapping seven-day rolling-origin test blocks and averages
three seeds within each fold before inference. Its +7-day placebo preserves event locations
and metadata. A null or negative A-versus-A+raw result remains the thesis result.

---

## 1. Research framing

- **Question.** Does adding *planned-event* information to a machine-learning traffic
  predictor improve accuracy, especially during event-driven congestion?
- **Hypothesis.** Event features (location, time, expected popularity) improve prediction
  accuracy, most visibly where congestion is externally driven by events.
- **Forecast task.** Measurements available through *t−15 minutes* predict the target for
  interval *t*. Planned-event metadata is assumed published before prediction.
- **Confirmatory method.** Hold the model, rows, split, seed, and hyperparameters fixed and vary
  only the feature set:
  - **Run A (baseline):** historical/temporal traffic features only.
  - **Run A+raw (event-aware):** the same features plus raw planned-event exposure fields.
  - A versus A+raw is the sole primary comparison. Other feature groups, Model B, alternate
    targets, and cross-city transfer are ablations or exploratory analyses.
- **Model B diagnostic.** Its residual target is an exploratory proxy also affected by
  incidents, weather, sensor noise, and omitted events; it is not the primary treatment.

---

## 2. Design decisions and rationale

Each subsection is a *decision point* you can cite/justify in the methodology chapter.

### 2.1 Target variable = traffic *flow* (volume), configurable to speed/occupancy
Predicting flow (vehicles per interval per sensor) matches loop-detector benchmark data
(Caltrans PeMS / PEMS-BAY) [7, 8] and the abstract's framing. Flow is a continuous value →
a **regression** problem, evaluated with MAE / RMSE / WAPE / R² and p95 absolute error [9].

*Caveat and configurability (real data).* Flow is **non-monotonic** in congestion: as a road
approaches gridlock, flow first rises then *falls* (fewer vehicles pass a jammed point), so a
large flow value is ambiguous (free-flowing vs. recovering). PeMS also reports **speed** and
**lane occupancy**, both of which are more monotonic congestion/"travel-time" proxies (speed
falls, occupancy rises, steadily into a jam). The pipeline therefore exposes `data.target ∈
{flow, speed, occupancy}` — the whole feature/lag/normalization/metric path follows the chosen
column — so the flow-vs-speed-vs-occupancy question can be studied empirically. Default stays
`flow` (matches the synthetic generator, which only simulates flow, and the abstract).

### 2.2 Real data by default, synthetic as a controlled testbed
The pipeline runs on **real data by default** (`data.source: real`): Caltrans PeMS loop-detector
flow (auto-fetched from a Google Drive mirror) + setlist.fm historical concerts (§2.12). A
**synthetic** generator is retained as an opt-in *controlled testbed* (`data.source: synthetic`),
and both write the identical schema so the model/feature code is source-agnostic.
- *Why the synthetic testbed is methodologically useful:* the generator injects a **known
  ground-truth event effect** into flow, so there is a measurable signal for Model B to learn
  and for the A-vs-A+ comparison to reveal. A model blind to events cannot explain the injected
  spikes; an event-aware one can. This establishes only that the implementation recovers its
  generator-injected effect under the generator's assumptions; it is not real-world evidence.
- *Its threat:* synthetic effect sizes reflect the generator's assumptions, not reality — hence
  real data is now the headline source, with synthetic as validation. See §5 and §6.

### 2.3 Data schema mirrors real sources (drop-in real data)
Schemas match real feeds so swapping in real data touches only the loaders, not the
feature/model code:
- `flow`  = `sensor_id, timestamp, flow` — the long form of a PeMS/PEMS-BAY flow matrix [7, 8].
- `sensors` = `sensor_id, lat, lon, base_flow` — detector metadata (fixed locations).
- `events` = `lat, lon, start_time, duration_h, category, expected_attendance` — the fields a
  concert feed provides; here **setlist.fm** [13] supplies the date/venue and a venue-capacity
  reference table supplies `lat/lon` + `expected_attendance` (= capacity as the popularity
  signal). See §2.12.

### 2.4 Storage format = Apache Parquet
Chosen for engineering reasons, **not** because the benchmarks use it (they don't — see the
honest note below):
- **Columnar + compressed:** far smaller and faster to read than CSV; a stage that needs
  only three columns reads only those columns [3, 4].
- **Preserves dtypes:** `timestamp` stays `datetime64`, ints stay ints — critical because
  the whole pipeline relies on datetime arithmetic; CSV would force re-parsing and risk
  silent type drift.
- **Standard in the pandas/PyArrow ecosystem** [4, 5], lossless round-trip.
- **Honest correction:** the widely used research releases METR-LA and PEMS-BAY are
  distributed as **HDF5** (`.h5`) [6], and raw Caltrans PeMS is CSV/text [7]. This project
  therefore *converts* incoming data to parquet internally for the reasons above; parquet
  is our processing choice, not the datasets' native format.

### 2.5 Primary A/A+raw design; Model B retained only as a diagnostic
The claim-bearing comparison bypasses the learned intermediate score. **A** receives only
traffic-history and temporal features; **A+raw** receives those identical columns plus the
predeclared raw event fields (`in_event_window`, distance, time-to-event, and
attendance-exposure fields). Holding the estimator, rows, fold, seed, and hyperparameters
fixed makes the event columns the only treatment difference.

Model B's learned `event_impact_score` remains an exploratory ablation. On synthetic data its
target can be the known injected event effect; on real data its proxy target is the baseline
model's positive residual. It is not used in the primary A-versus-A+raw result because an
unvalidated learned proxy would confound the test of whether planned-event metadata itself
helps.

### 2.6 Experimental control: same model type for A and A+
Both runs are built by the same `build_model()` and differ *only* in feature set. If the
model type also changed, a measured difference couldn't be attributed to the event feature.
This is standard controlled-experiment design.

### 2.7 Why two claim-bearing models (Random Forest and XGBoost)
Running the A-vs-A+raw test under two model families tests whether the result is a property
of the **data/features** or an artifact of one estimator. A consistent result across both is
stronger than one shown for a single model. Model choice rationale:
- **Random Forest** [1] — strong low-tuning tabular baseline; native feature importances
  let us *quantify* the event feature's contribution (useful thesis figure).
- **XGBoost** [2] — a strong boosted-tree robustness model with optional GPU training.
- Scikit-learn Gradient Boosting appears only in legacy exploratory outputs and is excluded
  from the verified six-cell claim-bearing run.
- On engineered-feature tabular problems, tree ensembles typically match or beat neural
  networks while needing far less tuning [11] — motivating trees as the primary family.

### 2.8 How models are compared (evaluation protocol)
- **Rolling-origin time splits**, not random: the verified run uses five non-overlapping
  seven-day test blocks. Each fold trains on all earlier data and never on its test block or
  the future, avoiding leakage through traffic autocorrelation.
- **Metrics** [9]: MAE, RMSE, MAPE (accuracy); plus max and 95th-percentile absolute error
  (reliability); plus inference latency in ms per 1 000 rows (the *response-time*
  criterion).
- **Two lenses:** metrics computed (a) over the whole test set and (b) over the
  **event-affected subset** (rows where the true event effect exceeds a threshold), because
  the hypothesis predicts the gain concentrates there. Reporting only the overall number
  would dilute and hide the effect.
- **Benchmark command** sweeps every model in `config.benchmark.models`, runs each as A and
  A+, and emits one comparison table + chart. Data prep and Model B run once, so only A/A+
  repeat per model. (On real data Model B's proxy target is derived from a baseline of the
  *base* config's `model.type`; the per-model runs share that one score, keeping the
  comparison consistent.)

### 2.9 Feature engineering choices
- **Lag features** (past target at t−1, −2, −3, −4, −96, −672 steps) — traffic is highly
  autocorrelated; recent, same-time-yesterday and same-time-last-week values are strong
  predictors. Lags are expressed in **timesteps and tuned to the data resolution**: at the
  15-minute PeMS/synthetic resolution, 96 steps = 1 day and 672 = 1 week (for hourly data the
  equivalents are 24 and 168). Computed on the configured target *within* each sensor so no
  series leaks into another; only past values used.
- **Rolling mean/std** (shifted by 1 so the window excludes the current step) — local trend
  and volatility.
- **Cyclical encoding** of hour/day-of-week via sin/cos — so 23:00 and 00:00 are adjacent
  to the model rather than maximally distant.
- **Event-exposure features** for A+raw and the Model B diagnostic (distance to nearest event,
  attendance-weighted and distance-decayed exposure, hours-to-next-event, in-event-window
  flag) — built only from event metadata + sensor location, i.e. information available before
  the forecast.

### 2.9a Anticipatory (lead-time) congestion — the core mechanism
The value of event data is *anticipatory*: traffic rises **before** an event because people
travel toward it (e.g. leaving an hour early for a match). A historical-only model only sees
congestion *after* it appears in the lag features — too late. An event-aware model knows a
large event starts soon and is nearby, so it can predict the rise **before any congestion
shows in the sensor data** — this is the abstract's "detect congestion earlier" claim made
concrete. Two features encode it:
- **`hours_to_next_event`** — a countdown to the nearest upcoming event (known in advance).
- **`pre_event_pressure`** — rises through the *approach window* `[start − lead, start]`,
  scaled by attendance and proximity, peaking at the start. `event_lead_hours` (default 3h,
  configurable) sets how early the approach begins.

The hypothesis therefore predicts the largest improvement on pre-event rows. The verified
real-data result does **not** show that improvement (§4.1b); this mechanism is observed only
in the controlled synthetic testbed (§4.1).

### 2.10 GPU / device handling
Only XGBoost is GPU-capable here (scikit-learn trees are CPU-only). A small detector picks
`cuda` when an NVIDIA GPU is present, else `cpu` (configurable/forceable). At this dataset
scale (~10^5 rows) GPU need not beat CPU — transfer/launch overhead dominates — so the
benchmark records inference latency to *show* the trade-off rather than assume a win.

### 2.11 Engineering hygiene (brief)
`uv` for reproducible environments; `black` for formatting; `mypy` for static typing;
`pytest` smoke test that runs the whole pipeline on a tiny config. Worth a sentence in a
"reproducibility" subsection.

### 2.12 Real-data integration (event source, timezone, missingness)
Turning the stubbed real-data path into a working one surfaced several data-quality decisions
worth documenting as methodology:

- **Historical event source = setlist.fm, not Ticketmaster.** Ticketmaster's Discovery API only
  lists *upcoming* events, so it cannot label a PeMS window that is already in the past.
  setlist.fm is a free, historical concert database. Its gaps — date only (no time), city-level
  coordinates, and no capacity — are filled by a **venue reference table** mapping each major
  LA/SF venue to precise `(lat, lon, capacity)`; **capacity is the popularity signal** (the
  hypothesis needs *expected* size, which venue capacity bounds, not measured turnout), the
  start time is assumed at a fixed local hour (20:00), and concerts at unlisted venues are
  dropped so every retained event has a trustworthy signal. Two further data-quality gates
  proved necessary: setlist.fm's venue-name search matches venues *nationwide* and several
  reference names are chains or duplicates (The Fillmore alone exists in half a dozen US
  cities; the Greek Theatre exists in both LA and Berkeley), so every returned setlist is
  validated against its own city coordinates (within 40 km of the reference venue) before it
  may inherit the venue's coords and capacity — without this, phantom out-of-state "events"
  contaminate the exposure features. And since setlist.fm returns one setlist *per performing
  artist*, multi-artist bills are collapsed to one event per venue-day so a headliner plus
  openers doesn't multiply the attendance signal. This is an honest-information
  argument: the model only ever sees what a real planner could know in advance. (A *live*
  deployment would instead pair an upcoming-events feed such as the Ticketmaster Discovery API
  [10] with a live traffic feed — see §5/§6; that is out of scope for this offline study.)
- **Timezone alignment.** PeMS timestamps are Pacific **local wall-clock with no timezone**;
  event feeds return UTC. Matching them naively puts events ~7–8 h off (an event would "cause"
  congestion in the middle of the night). Event times are converted into the PeMS local clock
  before feature construction. A silent version of this bug would have destroyed the event
  signal while leaving the code apparently working — a good cautionary methodology note.
- **Missing / imputed data — "un-filling", not infilling.** A dead loop detector does *not*
  mean "no traffic": the induction coil or its comms failed while cars kept driving over it.
  PeMS papers over such failures itself — every station reports a value every 5 minutes with
  no gaps (288/288 daily samples in the raw files), and dead detectors are backfilled with
  *typical* traffic for that time slot, flagged via `%Observed = 0`. The invented values look
  entirely plausible (mean 297 veh/5min vs 317 on live detectors in a sampled D7 day), so
  without the flag the imputation is invisible. The pipeline therefore does the *reverse* of
  infilling: bins whose average `%Observed` falls below a threshold (default 50 %) are set
  back to missing, kept on the regular 15-min grid so lag features stay time-aligned, and any
  row whose target or lag/rolling feature touches a missing value is dropped (listwise
  deletion). Two facts make this the right call for this thesis:
  1. *Imputed values cannot contain event spikes by construction* — they are historical
     averages, so training or scoring on them would systematically erase the event-driven
     deviations the study is trying to measure, biasing the experiment against its own
     hypothesis (and Model A would partly learn to predict PeMS's imputation model).
  2. *The deletion is unbiased here* — detectors die for hardware reasons unrelated to
     nearby events, so the missingness mechanism is independent of the quantity being
     studied and dropping those rows does not distort the event-effect estimate.
  Detector health turned out to be strongly **bimodal**: a station is either alive (≥99 % of
  its bins pass the gate) or dead (0 % observed, fully imputed) — there is almost no middle
  ground, so any threshold between ~20 % and ~80 % selects the same sensors. In the study
  window 86.6 % of D7 (LA) mainline stations and 39.3 % of D4 (SF) stations were dead; the
  gate excludes them rather than letting the model train on fabricated data. If naive
  resampling had also been left in place, empty bins would additionally have become
  *flow = 0* rows — fake "no traffic" — which the `n_obs` guard prevents.
- **Model B proxy target is cross-fitted.** With no ground-truth event effect on real data,
  Model B trains on the baseline's positive residual (traffic history couldn't explain). Using
  *in-sample* residuals makes the baseline look artificially good on its own training rows and
  shrinks the proxy signal; predictions are now **out-of-fold** (each train row scored by a
  baseline fit on the other folds), which is standard cross-fitting for a
  learned-target-from-a-model setup [14]. The folds are contiguous **time blocks**, not a
  shuffled K-fold — shuffling would predict each row with a baseline trained on rows
  interleaved with (and after) it, leaking the future through autocorrelation. Model B's own
  `event_impact_score` is cross-fitted the same way, so the diagnostic A+B ablation trains on
  out-of-fold scores of the same quality it will see on the test set rather than cleaner
  in-sample ones. The primary A+raw comparison does not use this score (§2.5).

### 2.13 Statistical rigor (significance of the A→A+ gap)
A single split + single seed cannot support "the event feature helps." `pipeline.evaluate`
adds: (i) five **rolling-origin** expanding-window folds with non-overlapping seven-day test
blocks; (ii) seeds 42, 43, and 44, averaged before inference; (iii) paired t/Wilcoxon tests and
a city-stratified day bootstrap for the MAE gap. The inferential unit is a paired city-day,
not an individual 15-minute sensor row or a seed replicate. The verified run contains 46
event-window city-days across 34 dates and 55 overall city-days across 35 dates. The reported
gap is **MAE(A) − MAE(A+raw)**, so positive favors event features and negative favors the
baseline.

---

## 3. How the models train (mental model)

**Tree ensembles do not have epochs** — epochs are a neural-network concept (repeated full
passes over the data updating weights by gradient descent). The two families used here
train differently:

- **Random Forest — bagging, parallel, independent trees** [1]. `n_estimators` trees are
  built at once. Each tree sees a random bootstrap sample of rows and a random subset of
  features at each split; it grows by choosing feature/threshold splits that most reduce
  prediction error. Final prediction = **average** of all trees. Picture *N* people each
  handed a different random slice of the data, each drawing their own flowchart; the answer
  is the crowd average. The data is not re-passed in an epoch sense.

- **Gradient Boosting / XGBoost — boosting, sequential trees** [2]. `n_estimators` = number
  of **rounds**, one tree per round, built one after another. Round 1 fits a weak tree and
  looks at where it's wrong (**residuals**); round 2 fits a tree to *those errors* and adds
  it, scaled by the `learning_rate`; round 3 fits the still-remaining errors; and so on.
  Each round shrinks the error a little. The learning rate controls how large each
  correction may be (smaller = slower, usually more accurate). This sequential
  error-correction is the closest analogue to "epochs," but the correct term is
  **boosting rounds**. (This is the loop that runs on the GPU for XGBoost.)

If a PyTorch neural net is added later (see §6), *that* model would genuinely train in
epochs and is where a GPU helps most.

---

## 4. Data and current results

### 4.0 The real dataset (for the paper's Data chapter)

Rebuilt and verified 2026-07-26 with all data-quality gates of §2.12 in place. Exact values
below come from `runs/real-20260725T210943Z/run_manifest.json`:

- **Traffic**: Caltrans PeMS Station 5-Minute, districts **D7 (Los Angeles)** and **D4 (SF
  Bay Area)**, window **2026-05-11 → 2026-07-11** (62 days), resampled to 15-minute bins
  (flow summed; speed/occupancy averaged), `%Observed ≥ 50 %` gating.
- **Events**: **164 concerts** (105 LA / 59 SF) at the 24 reference venues, after the venue
  city-validation and per-show dedup gates. (Before those gates the same fetch yielded 490
  "events" — roughly **two-thirds were contamination**: same-named venues elsewhere in the
  country, e.g. 93 phantom shows attributed to the SF Fillmore in 61 days, plus one setlist
  per performing artist per show. A useful cautionary tale for the methodology chapter.)
- **Sensors**: **954** retained mainline detectors near a reference venue (**473 LA, 481 SF**).
  These are counts in the post-gate sensor table, not a claim that every timestamp is usable.
- **Rows**: **5,678,208** regular 15-minute sensor-time rows before target missingness;
  **1,314,263** non-null observations are available for each of flow, speed, and occupancy
  (**23.14 %**). Lag and rolling-feature requirements reduce the fitted sample further.
- **Provenance**: input SHA-256 prefixes are `flow 64a63f6c…`, `events 53b74eb0…`, and
  `sensors 59263ee6…`; the run used Python 3.12.13, scikit-learn 1.9.0, and XGBoost 3.3.0
  on CPU.
- **Artifacts**: the authoritative files are under
  `runs/real-20260725T210943Z/`, not `media/results/`. This compact verified bundle and the
  `runs/latest-verified` pointer are explicitly versioned, so the results can be inspected
  and presented without rerunning model training; raw PeMS data and fitted models remain
  excluded.

### 4.1 Mechanism validation on the synthetic testbed

Verified sample run **`sample-20260716T165613Z`**, produced with:

```bash
uv run python -m scripts.run_sample_study --config config/sample.yaml
```

The generator created 42 days of 15-minute observations for 24 sensors and 90 planned
events across LA and SF: **96,768 rows**, of which **4,599 (4.8%)** carry a generator-injected
event effect above 0.05. The study completed **108 controlled fits in 619.3 seconds**:
Random Forest [1] and XGBoost [2] × three forward seven-day folds × three seeds × six treatments.
Within a fold and seed, every treatment uses identical rows, target, model family, and fixed
hyperparameters. Seeds are averaged before inference and are not counted as independent
observations.

Event-affected MAE by treatment (vehicles per 15-minute interval):

| Treatment | Added event information | Random Forest | XGBoost |
|---|---|---:|---:|
| A | None | 126.58 | 123.66 |
| A+window | Event-window indicator | 114.22 | 107.14 |
| A+spatiotemporal | Distance and time-to-event | 109.60 | 104.68 |
| A+attendance | Attendance-exposure features | 105.67 | 100.53 |
| **A+raw** | **All predeclared raw event features** | **103.79** | **97.06** |
| A+placebo | All metadata, event dates shifted +7 days | 126.50 | 123.85 |

The ablation pattern is monotonic in this run: the window alone helps, spatial/temporal and
attendance information help more, and A+raw has the lowest event-period MAE for both model
families. The date placebo performs approximately like traffic-only A, while correct event
timing reduces overall MAE relative to the placebo by 1.098 for Random Forest and 0.823 for
XGBoost. This supports the mechanism being tied to event timing rather than merely adding
event-shaped columns.

Uncertainty is based on paired city-day blocks with seeds averaged first. For event-affected
periods there are **23 city-day blocks across 18 distinct days**:

- Random Forest: mean paired A − A+raw MAE gap **+19.54**, stratified day-bootstrap interval
  **[+12.47, +26.57]**.
- XGBoost: mean paired A − A+raw MAE gap **+22.21**, stratified day-bootstrap interval
  **[+14.10, +30.31]**.
- Overall (42 city-day blocks across 21 days): Random Forest **+1.36**
  **[+0.76, +2.09]**; XGBoost **+1.25** **[+0.70, +1.89]**.

These intervals support improvement *under the synthetic generator's assumptions*. They do
not establish a real-world event effect or validate the assumed synthetic effect magnitude.
The appropriate claim is narrower: two different tree-model families recover the injected
signal, correct event dates outperform a +7-day placebo, and richer raw event information
progressively improves recovery. The real-data study determines the thesis conclusion.

Traceable artifacts are under `runs/sample-20260716T165613Z/`: `run_manifest.json` records
the input/configuration checksums, counts, versions, seed, hardware, timestamp, and Git commit;
`results/sample_results.json` contains both model families; and model-specific statistics are
in `results/stats_random_forest_flow.json` and `results/stats_xgboost_flow.json`.

### 4.1b Verified real-data results (headline)

Claim-bearing run **`real-20260725T210943Z`** completed on 2026-07-26. It evaluates flow,
speed, and occupancy with Random Forest and XGBoost. Each target/model cell uses five
forward seven-day folds × three seeds × six treatments: **90 controlled fits per cell and
540 fits overall**. Every treatment within a fold/seed uses the same rows and estimator
settings. The primary comparison is A versus **A+raw**; Model B is not in that comparison.

The table reports fold-averaged row-level event-window MAE for descriptive scale and the
paired city-day gap used for inference. Gap = MAE(A) − MAE(A+raw), so a negative value means
the event-aware model is worse. Intervals are 95 % city-stratified day-bootstrap intervals.

| Target | Model | Event MAE A | Event MAE A+raw | Paired event gap [95 % CI] | Event t-p / Wilcoxon-p | Paired overall gap [95 % CI] |
|---|---|---:|---:|---:|---:|---:|
| Flow | Random Forest | 66.174 | 75.177 | **−2.954 [−7.498, −0.385]** | 0.187 / 0.00261 | −0.256 [−0.788, +0.055] |
| Flow | XGBoost | 64.987 | 76.491 | **−3.915 [−9.706, −0.678]** | 0.162 / 0.00131 | −0.274 [−1.177, +0.270] |
| Speed | Random Forest | 2.223 | 2.325 | **−0.0823 [−0.1129, −0.0528]** | 6.24×10⁻⁶ / 4.12×10⁻⁶ | −0.0060 [−0.0094, −0.0030] |
| Speed | XGBoost | 2.228 | 2.300 | **−0.0618 [−0.0951, −0.0302]** | 0.000928 / 0.000519 | −0.0052 [−0.0084, −0.0024] |
| Occupancy | Random Forest | 0.01331 | 0.01483 | **−0.000670 [−0.001222, −0.000302]** | 0.0138 / 5.51×10⁻⁷ | −0.000039 [−0.000105, −0.000001] |
| Occupancy | XGBoost | 0.01330 | 0.01502 | **−0.000865 [−0.001461, −0.000383]** | 0.00543 / 7.94×10⁻⁶ | −0.000053 [−0.000109, −0.000013] |

**Conclusion.** No target/model cell supports the predeclared improvement claim. All six
event-window intervals are wholly negative, so raw planned-event metadata made predictions
worse during the periods where benefit was expected. Overall degradation is smaller:
the flow intervals include zero, while all speed and occupancy intervals remain slightly
negative.
The runner prints “interval does not support a directional claim” because its acceptance
flag tests only the positive/improvement direction; the negative intervals must still be
reported explicitly as evidence against improvement.

The +7-day placebo is also better than true event dates in every cell: the **overall**
A+raw-minus-placebo MAE is +0.741 (flow RF), +0.480 (flow XGBoost), +0.00264 (speed RF),
+0.00172 (speed XGBoost), +0.0000360 (occupancy RF), and +0.0000563 (occupancy XGBoost).
Negative would have favored the true dates. This result argues against the model extracting
a correctly timed concert signal.

On real data, “event-affected” means `in_event_window == 1`: a sensor is near an event and
the timestamp falls inside the configured approach/event window. It is not a label of
realized causal impact. The inferential sample contains 46 paired city-days across 34 dates
for this lens; overall inference uses 55 city-days across 35 dates.

**Thesis interpretation.** The controlled synthetic study shows the implementation can
recover a known injected effect (§4.1), but the real study rejects the expected benefit for
this data and feature design. Plausible explanations are incomplete concerts-only coverage,
date-only events assigned a fixed 20:00 start, venue capacity rather than attendance,
freeway mainline sensors that may miss venue-access traffic, the short 62-day window, and
strong lag features that leave little predictable event-specific residual. These are
limitations and follow-up hypotheses, not demonstrated causes.

### 4.1c Figures (generated by `scripts.run_visuals` → `media/figures/`)

Each figure is produced by a self-contained function in `src/pipeline/visualize.py` (usable
as a code snippet in the thesis). Suggested placement in the paper:

| File | Shows | Use in chapter |
|------|-------|----------------|
| `flow_profile.png` | Mean flow by hour, weekday vs weekend (rush-hour double peak) | Data description |
| `sensor_event_map.png` | Spatial layout of sensors + events (sized by attendance) | Data / setup |
| `event_effect_hist.png` | Distribution of injected event effect | Data / methodology |
| `feature_importance.png` | Which features the A+ model relies on (event score highlighted) | Results / interpretation |
| `pred_vs_actual.png` | Predicted vs actual flow for A and A+ | Results |
| `error_vs_event_effect.png` | **Key figure** — error vs event size; A+ pulls ahead as events grow | Results (hypothesis) |
| `xgb_training_curve.png` | XGBoost RMSE vs boosting rounds (visualizes §3 boosting) | Methodology (how models train) |
| `comparison_bars.png` | A vs A+ MAE/RMSE, overall and event-affected | Results |
| `benchmark.png` | Event-affected MAE, A vs A+, per model | Results (cross-model) |
| `event_window_sensor.png` | Actual vs A vs A+ around one real event | Results (qualitative) |

Note: `event_effect_hist.png`, `error_vs_event_effect.png`, and `event_window_sensor.png`
need the ground-truth event effect, so they are only produced on the **synthetic** testbed
(the run skips them on real data and says so). The full synthetic set (all ten figures) was
generated 2026-07-13 into `media/figures/synthetic/`; the real-data figures live in
`media/figures/`. Use the synthetic versions for the methodology/mechanism chapter and the
real ones for the results chapter.

### 4.2 Verified treatment ablations

These are descriptive fold-averaged event-window MAEs from the same verified run. Lower is
better. A+raw is the primary treatment; the intermediate columns test which subsets of raw
metadata help, and A+placebo shifts every event date by seven days.

| Target | Model | A | A+window | A+spatiotemporal | A+attendance | A+raw | A+placebo |
|---|---|---:|---:|---:|---:|---:|---:|
| Flow | Random Forest | **66.174** | 76.001 | 77.143 | 70.644 | 75.177 | 66.496 |
| Flow | XGBoost | **64.987** | 74.078 | 75.769 | 75.138 | 76.491 | 65.185 |
| Speed | Random Forest | **2.223** | 2.244 | 2.307 | 2.284 | 2.325 | 2.238 |
| Speed | XGBoost | **2.228** | 2.254 | 2.275 | 2.304 | 2.300 | 2.230 |
| Occupancy | Random Forest | **0.01331** | 0.01464 | 0.01506 | 0.01411 | 0.01483 | 0.01339 |
| Occupancy | XGBoost | **0.01330** | 0.01423 | 0.01460 | 0.01507 | 0.01502 | 0.01341 |

Baseline A is best in every target/model cell (XGBoost speed differs from placebo only in
the fourth decimal place but A remains lower). There is no monotonic improvement as richer
event information is added, and the correctly dated A+raw treatment is worse than its
+7-day placebo in all six cells. The former July 13 `split × normalize × model` table is a
legacy exploratory run and must not be combined with or substituted for these values.

---

## 5. Limitations / threats to validity

- **Synthetic effect sizes.** The synthetic testbed's effect sizes reflect the generator's
  assumptions (Gaussian rush peaks, distance-decayed event uplift), not measured reality —
  use them only as mechanism validation (§4.1), with real data as the headline (§4.1b).
- **No realized event-impact label.** The real dataset identifies scheduled event windows,
  not whether a particular event actually changed traffic at a particular detector. The
  primary A+raw result avoids Model B's learned proxy, but it still cannot distinguish a
  truly impacted sensor-time from a merely nearby/in-window one.
- **Concerts only — unlabeled events.** setlist.fm covers concerts; sports games, festivals,
  and conventions at the same venues (SoFi, Dodger Stadium, Chase Center host all of these)
  are *absent from the event table*. Their traffic still appears in the sensor data as
  unexplained congestion for **both** A and A+raw. This outcome misclassification can dilute
  the ability to detect a useful concert signal, but it does not justify calling the observed
  negative gaps a hidden gain.
- **Event metadata assumptions.** Start time is assumed 20:00 local (setlist.fm is
  date-only) and duration 3 h; venue **capacity** stands in for expected attendance (an upper
  bound on turnout, but the right "known in advance" signal — §2.12). Venue coverage is the
  24-entry reference table; concerts elsewhere are dropped.
- **Detector availability.** After `%Observed` gating, only **1,314,263 of 5,678,208**
  sensor-time rows (23.14 %) have a non-null target. The manifest's 473 LA / 481 SF sensor
  counts describe retained sensor metadata, not complete detectors; temporal availability and
  lag construction make the effective fitted sample smaller.
- **Event-affected lens on real data.** Without ground truth, the subset is defined by the
  event *window* (`in_event_window`), not by realized impact — a coarser lens that dilutes
  the per-row gain relative to the synthetic threshold lens (§4.1b).
- **Lag semantics vs resolution.** *Addressed.* Lags are now expressed in timesteps and tuned
  to the data resolution (`[1,2,3,4,96,672]` at 15-min → up to 1 h, 1 day, 1 week); §2.9.
- **Single seed / no significance test.** *Addressed* by `scripts.run_stats`
  (rolling-origin CV × seeds + paired significance test and bootstrap CI on the A→A+ gap;
  §2.13). The headline tables now report that claim-bearing run directly.
- **No spatial model.** Sensors are treated independently (no road-network topology); real
  traffic propagates spatially — a known gap vs spatio-temporal models [6].

---

## 6. To be done (future work)

1. **Real data.** *Done* (verified 2026-07-26) — all six predeclared target/model cells
   completed (§4.1b, §4.2). Outcome: raw event metadata does not improve prediction and
   worsens event-window MAE under the tested design.
2. **Raw-feature and alternate-target checks.** *Done*: the primary comparison now feeds raw
   event features directly into A+raw, and flow, speed, and occupancy were all evaluated.
   All six target/model intervals are negative. The next high-leverage data checks are to
   map retained detectors to actual venue approach routes, obtain precise event times and
   realized attendance, add sports/festival/convention coverage, and extend the date window.
3. **Statistical rigor.** *Done* — rolling-origin CV × seeds + paired significance test and
   bootstrap CI (`scripts.run_stats`, §2.13).
4. **Retune features** to the data resolution (lags, rolling windows). *Done* — lags are now
   resolution-tuned and config-driven (§2.9).
5. **Hyperparameter search** per model (e.g. XGBoost depth/learning-rate) for a fair
   best-vs-best comparison.
5a. **Live deployment.** This study is offline (archived PeMS + historical concerts). A real-time
   system would need a live event feed (e.g. Ticketmaster Discovery API [10]) *and* a live
   traffic feed — for which there is no free per-sensor-flow equivalent of PeMS: Google Maps
   exposes only travel-time-in-traffic (Routes API, paid), while TomTom/HERE offer real-time
   segment speed on freemium tiers. Adapting the model to one of those feeds is future work.
6. **Response-time study.** Formal CPU-vs-GPU latency and training-time comparison across
   dataset sizes (ties to the abstract's response-time criterion).
7. **Optional stretch: deep models.** A PyTorch MLP/LSTM (true epochs, GPU-bound), or a
   spatio-temporal GNN (DCRNN/STGCN) [6] that models the road network — higher risk, higher
   ceiling.

*(Done: the XGBoost training-curve figure from §3 is now generated as
`media/figures/xgb_training_curve.png` by `scripts.run_visuals`.)*

---

## 7. References

> Locate and reformat to your citation style; verify exact pages/DOIs before submission.

1. L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
2. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM
   SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, 2016, pp. 785–794.
3. Apache Software Foundation, "Apache Parquet" (columnar storage format), documentation,
   https://parquet.apache.org/ (accessed 2026).
4. Apache Software Foundation, "Apache Arrow / PyArrow" documentation,
   https://arrow.apache.org/ (accessed 2026).
5. W. McKinney, "Data Structures for Statistical Computing in Python," in *Proc. 9th Python
   in Science Conf. (SciPy)*, 2010, pp. 56–61. (pandas)
6. Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion Convolutional Recurrent Neural Network:
   Data-Driven Traffic Forecasting," in *Int. Conf. Learning Representations (ICLR)*, 2018.
   (Introduces the METR-LA and PEMS-BAY benchmarks, distributed as HDF5.)
7. California Department of Transportation, "Caltrans Performance Measurement System
   (PeMS)," https://pems.dot.ca.gov/ (accessed 2026). (Raw loop-detector flow, CSV/text.)
8. Original PeMS-based dataset construction for PEMS-BAY — see [6].
9. Standard regression metrics (MAE, RMSE, MAPE, R²); implementation via scikit-learn:
   F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine
   Learning Research*, vol. 12, pp. 2825–2830, 2011.
10. Ticketmaster, "Discovery API" documentation,
    https://developer.ticketmaster.com/ (accessed 2026). (Event location/time/venue feed.)
11. L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform
    deep learning on tabular data?," in *Advances in Neural Information Processing Systems
    (NeurIPS), Datasets and Benchmarks Track*, 2022.

13. setlist.fm, "setlist.fm API (v1.0)," https://api.setlist.fm/docs/1.0/ (accessed 2026).
    (Historical concert database; supplies event date + venue for the training-window events.)

14. V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and
    J. Robins, "Double/Debiased Machine Learning for Treatment and Structural Parameters,"
    *The Econometrics Journal*, vol. 21, no. 1, pp. C1–C68, 2018. (Cross-fitting /
    out-of-fold prediction when a model's output feeds another model, as in §2.12.)

*Optional / if used later:*
12. G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in
    *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

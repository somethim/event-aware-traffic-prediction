# Thesis reference notes — Event-Aware Traffic Prediction

Working notes to draw on when writing the thesis paper. Structured as: design decisions
(with rationale), how the models train, current results, limitations, future work, and
references. Reference markers like `[1]` point to the **References** section at the end.

> Citation caveat: bibliographic entries below give authors / title / venue / year so you
> can locate each source. **Verify exact page numbers, DOIs, and access dates before final
> submission** and reformat to your required citation style (IEEE / APA / etc.).

> **Result status (2026-07-16):** legacy files under `media/results/` are stale and are not
> claim-bearing. A result is verified only under `runs/<run_id>/`, with matching checksums in
> `run_manifest.json` and `runs/latest-verified` pointing to it. Final dataset counts must be
> rendered from that manifest, never maintained manually here.

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

### 2.5 Two-stage architecture (Model A/A+ + Model B)
Separating "predict traffic" from "quantify event pressure" keeps the comparison clean:
Model A/A+ never sees raw event rows, only Model B's single distilled score. This isolates
the event contribution to exactly one feature, so any A→A+ gain is attributable to it.
- *Model B target:* on synthetic data it trains on the ground-truth event effect; for real
  data the documented approach is a **proxy target** = the baseline model's fractional
  residual on event days (actual/predicted − 1, floored at 0), since events add traffic.

### 2.6 Experimental control: same model type for A and A+
Both runs are built by the same `build_model()` and differ *only* in feature set. If the
model type also changed, a measured difference couldn't be attributed to the event feature.
This is standard controlled-experiment design.

### 2.7 Why *multiple* models (RF, Gradient Boosting, XGBoost)
Running the A-vs-A+ test under several model families tests whether the benefit is a
property of the **data/features** or an artifact of one estimator. A gain that appears
across RF, GB, and XGBoost is a **stronger, more generalizable** claim than one shown for a
single model. Model choice rationale:
- **Random Forest** [1] — strong low-tuning tabular baseline; native feature importances
  let us *quantify* the event feature's contribution (useful thesis figure).
- **Gradient Boosting / XGBoost** [2] — often top performers on tabular data; XGBoost adds
  GPU training.
- On engineered-feature tabular problems, tree ensembles typically match or beat neural
  networks while needing far less tuning [11] — motivating trees as the primary family.

### 2.8 How models are compared (evaluation protocol)
- **Time-based split**, not random: the last 20 % of the timeline is the test set, so the
  model never trains on the future — essential for an honest forecasting evaluation
  (random splits leak future information via autocorrelation).
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
- **Event-exposure features** for Model B (distance to nearest event, attendance-weighted
  and distance-decayed exposure, hours-to-next-event, in-event-window flag) — built only
  from event metadata + sensor location, i.e. exactly what a real scraper yields, keeping
  the information content honest.

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

This is why the event feature helps *most* on the pre-event rows, and why it can lower error
earlier than lag-only models — the key thing to highlight in the results discussion.

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
  `event_impact_score` is cross-fitted the same way, so run A+ trains on out-of-fold scores of
  the same quality it will see on the test set rather than cleaner in-sample ones.

### 2.13 Statistical rigor (significance of the A→A+ gap)
A single split + single seed cannot support "the event feature helps." `pipeline.evaluate`
(`scripts.run_stats`) adds: (i) **rolling-origin** (expanding-window) time-series CV — several
forward test blocks, never training on the future; (ii) **multiple seeds** per fold to average
out estimator randomness; (iii) a **paired significance test** (t-test + Wilcoxon backup) and a
**bootstrap confidence interval** on the per-(fold, seed) MAE gap, reported separately for the
event-affected subset (the primary lens) and overall. The claim becomes "A+ beats A by *X* ±
CI, p = …" rather than a single point estimate.

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

Built 2026-07-12 with all data-quality gates of §2.12 in place:

- **Traffic**: Caltrans PeMS Station 5-Minute, districts **D7 (Los Angeles)** and **D4 (SF
  Bay Area)**, window **2026-05-11 → 2026-07-11** (62 days), resampled to 15-minute bins
  (flow summed; speed/occupancy averaged), `%Observed ≥ 50 %` gating.
- **Events**: **164 concerts** (105 LA / 59 SF) at the 24 reference venues, after the venue
  city-validation and per-show dedup gates. (Before those gates the same fetch yielded 490
  "events" — roughly **two-thirds were contamination**: same-named venues elsewhere in the
  country, e.g. 93 phantom shows attributed to the SF Fillmore in 61 days, plus one setlist
  per performing artist per show. A useful cautionary tale for the methodology chapter.)
- **Sensors**: 954 mainline detectors within `event_radius_km + station_buffer_km` (5+3 km)
  of a reference venue. Detector health is bimodal (§2.12): 54 LA + 82 SF sensors have ≥80 %
  usable bins and contribute **~695 k fully-usable rows** (300 k LA / 394 k SF); the rest sit
  under dead detectors and drop out in feature construction (23.1 % of all bins usable).
- Result artifacts land in `media/results/` (`metrics.json`, `benchmark.json`,
  `experiments.json`, `stats.json`) and `media/figures/`; the experiment-matrix table in
  §4.2 is rewritten automatically by `scripts.run_experiments` / `scripts.run_all`.

> ✅ **Regenerated 2026-07-13**: `run_all` + `run_stats` completed on the cleaned dataset;
> §4.1b and §4.2 below hold the numbers from that run. (Real-data numbers produced before
> 2026-07-12 remain invalid — contaminated events table, shuffled cross-fitting — do not
> quote them.)

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

### 4.1b Real-data results (headline) — a null result, honestly reported

Run of 2026-07-13 on the cleaned dataset (§4.0), `run_all` + `run_stats`. The short version:
**on real data the event feature does not improve prediction.** The synthetic testbed (§4.1)
shows the *mechanism* works when a real event signal exists; the real-data experiment shows
the current pipeline fails to extract such a signal from concerts + freeway sensors. The
evidence, in causal order:

- **Model B learns nothing.** In every condition of the experiment matrix, Model B's fit on
  its cross-fitted proxy-residual target is **R² ≈ 0.00** (range −0.001 to +0.001). The
  `event_impact_score` handed to A+ is therefore approximately noise, and every downstream
  A-vs-A+ delta must be read in that light. This is the root cause of everything below.
- **Headline comparison** (default RF, time split; `metrics.json`): overall MAE
  57.96 → 57.91 (**+0.10 %**), RMSE 90.64 → 90.53, R² = 0.970 for both runs. On the 3,570
  event-window rows (6.4 % of the 55,746 test rows): MAE 74.99 → 75.15 (**−0.22 %**) — A+
  is slightly *worse* exactly where the hypothesis predicts it should be better.
- **Significance run** (`stats.json`; rolling-origin CV, 4 folds × 3 seeds, RF; gap = A − A+
  so positive = A+ better):
  - OVERALL: mean MAE gap **+0.001** [−0.42, +0.45], t-p = 0.996 — indistinguishable from 0.
  - EVENT-AFFECTED: mean MAE gap **−4.23** [−8.56, −0.19], t-p = 0.074, Wilcoxon p = 0.009.
    The negative sign means A+ is on average *worse* on event rows. The printed verdict is
    "not significant" only because the criterion is one-directional (requires an
    improvement CI); read honestly, there is no evidence of benefit and weak-to-moderate
    evidence of harm on the event subset.
  - The event-subset harm is driven almost entirely by **fold 1**, where A scores ≈ 48.6
    MAE and A+ jumps to 64–67 across all three seeds; the other three folds sit near zero
    gap. Before writing the discussion chapter, diagnose fold 1 (which dates it covers,
    which events fall in it) — a noise feature that occasionally *destabilizes* the model
    is itself a reportable failure mode.
- **MAPE is unusable on this data** (values in the 10⁶–10⁹ % range): near-zero night-time
  flow puts ≈0 in the denominator. Quote MAE / RMSE / R² and drop MAPE from real-data
  tables, or replace it with WAPE / sMAPE.
- `benchmark.json` was **not** regenerated in this run — do not quote it next to these
  numbers.

Note on the lens: on real data "event-affected" means `in_event_window == 1` (inside
`[start − lead, end + 1 h]` of a nearby event) — a coarser lens than the synthetic
ground-truth threshold, since there is no true label (§2.5). It dilutes a real gain rather
than inflating one, but it cannot explain a *negative* event-window delta.

**How to use this in the thesis.** Yes, this is usable — a null result with a diagnosed
cause is a legitimate, defensible contribution; do not spin it. The clean framing is a
two-part story:

1. *Mechanism validation (synthetic, §4.1):* when a ground-truth event effect exists in the
   data, the two-stage architecture detects and exploits it (+15–20 % on event rows across
   three model families, significant at p < 0.001 under the same CV × seeds protocol, on a
   testbed matched to the real dataset's window, event density, and flow scale). The design
   is sound.
2. *Real-data finding:* with concerts-only event coverage, a coarse window lens, freeway
   mainline sensors, a 62-day window, and a proxy target the impact model cannot learn
   (R² ≈ 0), the event feature adds no measurable value overall and may destabilize
   event-window predictions. The gap between (1) and (2) localizes the failure to the
   *signal-extraction* step (Model B's target), not the architecture — which directly
   motivates the future-work items in §6.

Candidate explanations for why the proxy target is unlearnable, for the discussion chapter:
the residual of a strong lag model is mostly irreducible noise; concerts may barely move
*freeway mainline* flow (attendees load arterials and parking streets near venues, which
PeMS mainline detectors don't see); 164 events in 62 days is a thin training signal; and
the assumed 20:00 start time / 3 h duration blurs the true event windows.

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

### 4.2 Experiment matrix

Every combination of `split × normalize × model` is run separately by
`scripts.run_experiments`; the table below is auto-generated (re-run to refresh). It shows
which configuration gives the largest event-affected accuracy gain, per condition.

<!-- EXPERIMENTS:START -->

| split | normalize | model | MAE A | MAE A+ | Δ% overall | MAE A (event) | MAE A+ (event) | **Δ% event** | infer ms/1k |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| time | none | random_forest | 57.96 | 57.91 | +0.10% | 74.99 | 75.20 | **-0.28%** | 28.71 |
| time | none | gradient_boosting | 59.87 | 59.76 | +0.18% | 75.34 | 75.19 | **+0.21%** | 1.87 |
| time | none | xgboost | 56.76 | 56.82 | -0.09% | 72.91 | 73.06 | **-0.20%** | 0.52 |
| time | per_sensor | random_forest | 57.81 | 57.80 | +0.01% | 73.99 | 74.43 | **-0.59%** | 29.60 |
| time | per_sensor | gradient_boosting | 59.28 | 59.24 | +0.07% | 75.51 | 75.50 | **+0.01%** | 2.01 |
| time | per_sensor | xgboost | 56.94 | 57.22 | -0.50% | 73.05 | 72.82 | **+0.32%** | 0.51 |
| city | none | random_forest ⭐ | 57.86 | 56.33 | +2.65% | 87.39 | 83.05 | **+4.97%** | 15.51 |
| city | none | gradient_boosting | 58.07 | 57.05 | +1.75% | 70.21 | 75.58 | **-7.65%** | 2.95 |
| city | none | xgboost | 56.75 | 56.51 | +0.41% | 76.26 | 84.64 | **-11.00%** | 0.26 |
| city | per_sensor | random_forest | 59.29 | 57.67 | +2.73% | 88.56 | 87.63 | **+1.05%** | 18.94 |
| city | per_sensor | gradient_boosting | 56.68 | 55.47 | +2.14% | 67.78 | 75.95 | **-12.05%** | 2.67 |
| city | per_sensor | xgboost | 54.66 | 55.56 | -1.64% | 77.19 | 92.41 | **-19.71%** | 0.25 |

**Best configuration per condition** (largest event-affected MAE reduction):

- *Within-city (time split):* `xgboost` + normalize=`per_sensor` → +0.32% on event rows.
- *Cross-city transfer (LA→SF):* `random_forest` + normalize=`none` → +4.97% on event rows.

Δ% = MAE reduction from baseline A to event-aware A+ (higher is better). The event feature helps most on event-affected rows; ⭐ = best cell overall.

<!-- EXPERIMENTS:END -->

**Reading the matrix honestly (2026-07-13 run).** The time-split event deltas (−0.59 % to
+0.32 %) are within run-to-run noise. The city-split deltas swing from +4.97 % to −19.71 %
and flip sign across models and normalizations — that is instability of the LA→SF transfer
setting, not an event effect: an uninformative extra feature (Model B's R² ≈ 0, §4.1b)
perturbs each estimator differently under distribution shift. Note also the mixed pattern
that overall MAE often *improves* under the city split while event-window MAE worsens — the
opposite of the hypothesis, which predicts the gain concentrates on event rows. No cell in
this table supports a benefit claim; do not quote the ⭐ cell in isolation (the boilerplate
caption below the table is auto-generated and predates this finding).

---

## 5. Limitations / threats to validity

- **Synthetic effect sizes.** The synthetic testbed's effect sizes reflect the generator's
  assumptions (Gaussian rush peaks, distance-decayed event uplift), not measured reality —
  use them only as mechanism validation (§4.1), with real data as the headline (§4.1b).
- **Model B proxy target failed to validate on real data.** On synthetic data Model B
  trains on the true effect; real data has no such label, and the cross-fitted
  proxy-residual target (§2.5, §2.12) turned out to be **unlearnable in practice**:
  R² ≈ 0.00 in every experiment condition (§4.1b). The `event_impact_score` is therefore
  ~noise on real data, which caps everything A+ can show and is the diagnosed cause of the
  real-data null result. See §4.1b for candidate explanations and §6 for what to try next.
- **Concerts only — unlabeled events.** setlist.fm covers concerts; sports games, festivals,
  and conventions at the same venues (SoFi, Dodger Stadium, Chase Center host all of these)
  are *absent from the event table*. Their traffic still appears in the sensor data as
  unexplained congestion for **both** A and A+, which adds noise and — because A+ gets no
  feature for them either — should *dilute*, not inflate, the measured event gain. State this
  direction-of-bias argument explicitly: the reported gain is a lower bound w.r.t. event
  coverage.
- **Event metadata assumptions.** Start time is assumed 20:00 local (setlist.fm is
  date-only) and duration 3 h; venue **capacity** stands in for expected attendance (an upper
  bound on turnout, but the right "known in advance" signal — §2.12). Venue coverage is the
  24-entry reference table; concerts elsewhere are dropped.
- **Detector coverage (LA).** 86.6 % of D7 mainline stations were dead (fully imputed) in the
  study window, leaving ~54 healthy LA sensors vs ~82 for SF (§2.12, §4.0). The %Observed gate
  is the honest choice, but it means the LA spatial coverage is thin and the effective dataset
  is smaller than the raw row counts suggest — report the healthy-sensor counts, not the 954.
- **Event-affected lens on real data.** Without ground truth, the subset is defined by the
  event *window* (`in_event_window`), not by realized impact — a coarser lens that dilutes
  the per-row gain relative to the synthetic threshold lens (§4.1b).
- **Lag semantics vs resolution.** *Addressed.* Lags are now expressed in timesteps and tuned
  to the data resolution (`[1,2,3,4,96,672]` at 15-min → up to 1 h, 1 day, 1 week); §2.9.
- **Single seed / no significance test.** *Addressed* by `scripts.run_stats`
  (rolling-origin CV × seeds + paired significance test and bootstrap CI on the A→A+ gap;
  §2.13). The headline tables are single default runs for readability; the stats run is the
  claim-bearing evidence.
- **No spatial model.** Sensors are treated independently (no road-network topology); real
  traffic propagates spatially — a known gap vs spatio-temporal models [6].

---

## 6. To be done (future work)

1. **Real data.** *Done* (2026-07-13) — dataset built (§4.0) and all numbers regenerated
   (§4.1b, §4.2). Outcome: a null result on real data with a diagnosed cause; §4.1b gives
   the two-part framing for the thesis.
2. **Validate Model B's proxy target** on real event days. *Done — validation failed*:
   R² ≈ 0 in every condition (§4.1b, §5), so the score does not track real event uplift.
   If a positive real-data result is wanted, the highest-leverage next steps are:
   (a) drop the two-stage score and feed the raw event features (distance to venue,
   hours-to-event, attendance, pre-event pressure) **directly into A+** — this removes the
   unlearnable intermediate target entirely; (b) verify event/sensor overlap in space and
   time (do any healthy sensors sit on venue approach routes?); (c) try
   `data.target: speed` or `occupancy`, where event effects may be more visible than in
   mainline flow; (d) diagnose the fold-1 instability from the stats run (§4.1b).
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

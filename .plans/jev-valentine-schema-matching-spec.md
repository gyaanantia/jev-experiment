# Spec: Jev as a schema matcher on the Valentine benchmark

Status: interview complete, 2026-10-30. Talk is in 3 days (about 2026-11-02).
Owner: gyaan@qualgent.ai. Project root: `/Users/gyaanantia/L2` (standalone, runs on this Mac).

## 1. Goal

Measure how well Jev, TypeSafe's System One model, matches columns between two tables. Compare it with the methods in *Valentine: Evaluating Matching Techniques for Dataset Discovery* (Koutras et al., ICDE 2021). Report accuracy, cost, and latency in one interactive HTML page that the presenter screen-shares during a talk to DB/ML researchers. Every number on the page links to its source.

**How Jev matches columns.** For each source column and each target column, the code asks Jev one yes/no question (a *Noul*): "Do these two columns hold the same attribute?". Jev returns `noul`, the probability that the answer is yes (0 to 1). The code ranks all column pairs by `noul`. This ranked list is the method's output, and it uses the same output format as every Valentine matcher.

## 2. Decisions from the interview

| Topic | Decision |
|---|---|
| Audience | DB/ML researchers. Show method details, variance, ablations, and fairness caveats. |
| Baseline numbers | Both: re-run today's `valentine` package on the same pairs on this Mac, **and** show the paper's published numbers with page/table citations. |
| Datasets | All public paper pairs plus the 10 NYU pairs. **Start with a small pilot, check that it works, then run everything.** |
| Jev input modes | Three: schema-only, instance-only, hybrid. Each mode is compared with the paper's matching method family. |
| Packing | Full run: one request per source column (fan-out). Pilot: also one request per column pair, to compare cost and latency and to check that the answers agree. |
| Column context | Column profile plus the names of the other columns in its table (see §5.2 for instance-only mode). |
| Metrics | Recall@ground truth (the paper's metric) + thresholded P/R/F1 (noul ≥ 0.5, ≥ 0.7) + MRR / P@1 per source column + one-to-one F1 (Hungarian). Calibration metrics were **not** selected. |
| Prompt tuning | Tune on a dev split, freeze, then run the rest once. Report dev pairs separately. |
| Question wording | Scenario-agnostic. Jev is never told whether the tables are unionable or joinable. |
| Ties | Expected-value Recall@GT. Pessimistic and optimistic bounds shown as error bars. |
| Model | Pin `jev-1.14.0`. Log the `model` field of every response. |
| Latency | Wall-clock per table pair (comparable to the paper's Table V) and per-request p50/p95, plus a concurrency sweep (1, 8, 32). |
| Sample size | Sweep 5 / 10 / 25 / 50 values on the dev split, then fix one value. |
| Stability | Run the dev split 3 times with the frozen prompt. |
| LLM baseline | Claude Haiku 5.5 with the Messages API on a stratified subset of about 60 pairs, hybrid mode. **User adds `ANTHROPIC_API_KEY` to `.env`.** |
| Baseline tuning | Grid search on the same dev split (ranges from paper Table II), freeze, run on all pairs with a per-pair timeout. |
| Time fallback | If the baselines cannot finish in about 36 h: run them on the subset (dev split + Haiku subset) and show the paper's values for the other pairs. |
| Statistics | Paired Wilcoxon tests with Holm correction, bootstrap 95% CIs, win/tie/loss counts, difficulty breakdown. |
| Budget: Jev | The code estimates cost from counted tokens before each phase and stops automatically at a cap. $5 now; the user can add $5 more. **The absolute cap is $10 in total.** |
| Budget: Haiku | Hard cap $5. |
| Report | One local, self-contained interactive HTML page (no Artifact). It is laid out in talk order, with keyboard navigation between sections. |
| Extras | Slide-ready PNG and SVG figures. |
| Live demo | Deferred. Build it only if there is time after the results are final. |
| Jetson Orin Nano | Not used. The baselines are CPU algorithms with no GPU path, and the i9 is faster (see §11). |

## 3. Data

### 3.1 Sources

| Source | Pairs | Where it comes from | Paper reference |
|---|---|---|---|
| TPC-DI (Prospect table) | 180 | Zenodo `Valentine-datasets.zip` | §V-A, p. 473 |
| Open Data | 180 | Zenodo | §V-A, p. 473 |
| ChEMBL (Assays table) | 180 | Zenodo | §V-A, p. 473 |
| WikiData | 4 | Zenodo (check that it is there) | §V-B, p. 473 |
| Magellan | 7 | Zenodo (check that it is there) | §V-B, p. 473; Table IV, p. 477 |
| NYU (NYC Open Data, LLM-perturbed) | 10 | `delftdata/valentine` repo, `experiments/data/`, pinned to a commit SHA | Not in paper; held-out set |
| ING | 2 | Private, not available | Excluded |

- Zenodo record: <https://zenodo.org/record/5084605>, file `Valentine-datasets.zip`, 602,150,347 bytes. Check the md5 checksum after the download.
- If WikiData or Magellan are not in the zip, look for them in the original experiment suite (repo tag `v1.1`). If they cannot be found, exclude them and say so in the report.
- Downloaded archives are untrusted data. Extract them into `data/raw/`, and run every Python script that reads them with `python -I`.

### 3.2 Pair index

`data/pairs.parquet` has one row per pair: `pair_id`, `source` (TPC-DI, …), `scenario` (unionable, view-unionable, joinable, semantically-joinable, or curated), `schema_noise` (verbatim/noisy), `instance_noise` (verbatim/noisy), `row_overlap`, `col_overlap`, `n_src_cols`, `n_tgt_cols`, `n_gt`, `split` (pilot/dev/test), `haiku_subset` (bool). Parse the scenario and noise flags from the Zenodo folder names, and check them against Fig. 3 of the paper.

### 3.3 Splits (seed 42, stratified)

- **Pilot (about 6 pairs):** one pair from each of TPC-DI, Open Data, ChEMBL, WikiData, Magellan, and NYU. Use the smallest pair of each where possible. The pilot checks that the pipeline works end to end. Pilot pairs stay in the test set (pilot answers are discarded and the pairs are run again with the frozen setup).
- **Dev (24 pairs):** 3 fabricated sources × 4 scenarios × 2 schema-noise levels, one pair per cell. Prompt wording, sample size, and baseline grids are tuned only on these pairs. Dev pairs are excluded from all headline numbers.
- **Test:** all other pairs (about 537). Headline numbers come only from test pairs.
- **Haiku subset (about 60 pairs, inside Test):** 2 pairs per fabricated source × scenario × schema-noise cell (48), plus all WikiData and Magellan test pairs and 1 NYU pair. This is also the baseline fallback subset (§9).

## 4. Methods compared

| Method | Family (paper) | Source of numbers |
|---|---|---|
| Jev schema-only | schema | this run |
| Jev instance-only | instance | this run |
| Jev hybrid | hybrid | this run |
| Haiku 5.5 hybrid | hybrid (LLM) | this run, Haiku subset only |
| Cupid | schema | re-run + paper Fig. 4, Table IV |
| Similarity Flooding | schema | re-run + paper Fig. 4, Table IV |
| COMA schema | schema | re-run (`use_schema=True, use_instances=False`) + paper |
| COMA instance | instance | re-run (`use_instances=True`) + paper Fig. 5, Table IV |
| Distribution-based | instance | re-run + paper Fig. 5, Table IV |
| Jaccard-Levenshtein | instance | re-run (`JaccardDistanceMatcher`, Levenshtein) + paper |
| SemProp, EmbDI | hybrid | **paper only**: not in today's package |

Use the latest `valentine` release (1.1.0 at time of writing). Record its exact version in the report. The package rewrote COMA in Python after 2021, so re-run numbers can differ from the paper. The report shows both values and states this.

## 5. Jev request design

### 5.1 Client

- Use `typesafe-sdk` (Python, currently 0.7.5) with `AsyncTypeSafeClient`, the endpoint `POST https://api.typesafe.ai/v1/systemone`, and `model="jev-1.14.0"`.
- Let the SDK handle retries (429/529 with backoff). Log every 429.
- The code loads `TYPESAFE_API_KEY` from `.env` with `python-dotenv`. **No script prints, logs, or writes the key.** Claude does not read `.env`.

### 5.2 Column profile (what Jev sees)

Each column becomes a JSON object. The fields depend on the mode:

| Field | schema-only | instance-only | hybrid |
|---|---|---|---|
| `name` | yes | no | yes |
| `inferred_type` (string/integer/float/date/bool) | yes | no (the values show it) | yes |
| `sample_values` (N distinct non-null values) | no | yes | yes |
| `other_columns_in_table` (names) | yes | no (names are hidden in this mode) | yes |

- **Sampling:** N distinct non-null values from the column's own table, chosen at random with a fixed seed. Each value is cut to at most 60 characters (the exact length is fixed in the pilot). N comes from the sweep (§7.2).
- **Table names are never sent.** Zenodo file names contain hints such as `source`/`target` and the scenario, which would leak information.
- Same rule as the Valentine package: no method gets information that identifies the dataset.

### 5.3 Fan-out packing (full run)

- One request per source column.
- `state` = `{"source_column": <profile>}`.
- `questions` = one `Noul` per target column. Each `Noul` has structured `instructions`: `{"candidate_column": <target profile>, "question": "<frozen wording>"}`.
- Question ids are code-generated (`t_0`, `t_1`, …) and mapped back to column names in code.
- Before sending, check that each request is under the 64k-token request limit and the 32k-token per-question limit. If a request is too large, split its questions across several requests.

Starting wording (iterated on the dev split, then frozen): *"Do `source_column` and `candidate_column` hold the same attribute of the same kind of entity, so that their values could be stored in one column?"* Optional `criteria` (true/false descriptions) are tested on the dev split.

### 5.4 Pilot-only packing comparison

On the pilot pairs, run both setups:
- **Singles:** one request per column pair.
- **Fan-out:** as in §5.3.

Report the cost ratio, the latency ratio, and the agreement between the two (mean |Δnoul|, Spearman ρ). On 2 pilot pairs, also run the reverse direction (target as state) to check symmetry. Only fan-out is used for the full run.

### 5.5 Precision of noul values

In the pilot, check how many decimals `noul` has. If it is rounded (for example to 2 decimals), many scores tie, and the tie rule in §8.1 applies.

## 6. Haiku 5.5 baseline

- Use the `anthropic` Python SDK with `model="claude-haiku-5-5"` and the key from `ANTHROPIC_API_KEY` (in `.env`).
- **Packing:** one request per source column, listing all candidate target columns. The model returns structured JSON (`output_config.format`) with one probability (0–1) per candidate.
- **Known difference:** Haiku judges all candidates together in one answer, while Jev judges each candidate on its own. The report states this.
- **Same inputs:** the same hybrid profiles, the same N, and the same question wording as Jev.
- **Settings:** decided on the dev split between thinking off at effort `low` and adaptive thinking at effort `low`, then frozen. No sampling parameters (`temperature` and similar are rejected since 2026-10-21).
- **Probabilities:** Haiku has no token probabilities. Its score is a stated probability, so ties are likely and the §8.1 tie rule applies.
- **Cost:** computed from `usage` × list price: $0.10 per Mtok input and $0.50 per Mtok output for prompts up to 100K tokens. Check the price against the live Anthropic pricing page before publishing, and cite it with the date it was read.
- **Latency:** wall-clock time of each Messages API call, with no CLI in between.

## 7. Phases

Each phase ends with a check. Cost estimates come from counted tokens, not from guesses.

### Phase 0 — Setup (day 1)
1. Create a `uv` project with Python 3.12. Dependencies: `typesafe-sdk`, `anthropic`, `valentine`, `pandas`, `pyarrow`, `scipy`, `numpy`, `python-dotenv`, `matplotlib` (for figure export).
2. Download and check the Zenodo zip. Fetch the NYU pairs at a pinned commit. Build `data/pairs.parquet`.
3. Send one Jev smoke-test request (costs less than $0.001) to confirm the key works.

### Phase 1 — Pilot (day 1)
1. Run the 3 Jev modes on the pilot pairs with fan-out. Also run singles, and the reverse direction on 2 pairs.
2. Run the concurrency sweep (1, 8, 32 requests in flight) on one Open Data pilot pair. Record 429s.
3. Run each of the 6 baselines with package defaults on the pilot pairs, to measure runtime per pair on this Mac.
4. **Output:** a check that metrics are computed correctly against hand-checked ground truth, the noul precision, tokens per question by mode and source, and **a cost and time estimate for every later phase.** If the Jev estimate goes over the remaining cap, ask the user to top up (absolute cap $10).

### Phase 2 — Dev split (day 1–2)
1. **Sample-size sweep:** N ∈ {5, 10, 25, 50} × 3 modes on the 24 dev pairs (instance and hybrid modes; schema-only has no N). Plot accuracy against cost.
2. **Prompt wording:** 2–4 variants (with and without `criteria`), hybrid mode. Pick the best by mean Recall@GT, with cost as the tie-breaker.
3. **Freeze** the wording, N, and value length. Write them to `config/frozen.json` with a timestamp.
4. **Stability:** run the frozen setup 3 times on the dev split. Report the per-question SD of noul, the share of identical answers, and the Recall@GT range.
5. **Baseline grids (in the background, in parallel):** use the Table II ranges on the dev split. Cupid: `w_struct`, `leaf_w_struct` ∈ [0, 0.6] step 0.2, `th_accept` ∈ [0.3, 0.8] step 0.1. Distribution-based: θ1, θ2 ∈ [0.1, 0.5]. Jaccard-Levenshtein: threshold ∈ [0.4, 0.8] step 0.1. COMA: max_n=0, threshold 0, as in the paper. Similarity Flooding: inverse average, formula C, as in the paper. Choose the best configuration by mean dev Recall@GT and freeze it. If the grid will not finish in time, shrink it and say so.
6. **Haiku dev check:** decide the thinking setting (§6) and freeze it.

### Phase 3 — Full runs (day 2)
1. Jev, 3 modes, all test pairs, frozen setup, at the highest concurrency that did not cause errors in Phase 1.
2. Haiku hybrid on the Haiku subset.
3. Baselines with frozen configurations on all test pairs, one pair per CPU core (8 workers), with a per-pair timeout (set from the pilot runtimes). **Check at about 36 h:** if the baselines will not finish, stop at the subset (§9).

### Phase 4 — Analysis and report (day 2–3)
Compute metrics (§8), build `report/index.html` (§10), and export the figures.

### Phase 5 — Optional live demo
Build it only if Phases 0–4 are done with time left. It would be a small local Python server that keeps the key on this Mac. Not planned in detail.

## 8. Metrics

All metrics are computed for each pair and then aggregated for each method × scenario (× noise) cell. One module computes them for every method, so every method uses the same code. Where possible, the results are cross-checked against `valentine`'s own `get_metrics`.

### 8.1 Recall@ground truth (headline)

Rank all m×n pairs by score. Let k = |GT|. Recall@GT = (number of GT pairs in the top k) / k, as defined on p. 470.

**Ties:** let t be the score at rank k, A the pairs scored above t, and T the pairs scored exactly t. The r = k − |A| remaining slots are filled from T.

- Expected (headline): (|A∩GT| + r·|T∩GT|/|T|) / k.
- Pessimistic: (|A∩GT| + max(0, r − |T∖GT|)) / k.
- Optimistic: (|A∩GT| + min(r, |T∩GT|)) / k.

### 8.2 Other metrics

- **Thresholded precision, recall, F1** at noul ≥ 0.5 and ≥ 0.7. Only for Jev and Haiku, because baseline scores are not probabilities.
- **MRR and P@1 per source column**, over the source columns that have a GT match.
- **One-to-one F1** after Hungarian assignment (the default of today's `valentine`), for every method.
- **Cost per pair (USD):** Jev = input_tokens × $0.042/Mtok (output tokens are free); Haiku = list price; baselines = $0 API cost (local CPU time is reported instead).
- **Latency:** wall-clock per pair, from the loaded CSVs to the ranked list; per-request p50 and p95; concurrency sweep.

### 8.3 Statistics

- Bootstrap 95% CIs (10,000 resamples over pairs) for mean and median Recall@GT, cost, and latency.
- Paired Wilcoxon signed-rank test, best Jev mode against each baseline, for each scenario, with Holm correction. Only pairs where both methods finished are used.
- Win/tie/loss counts against each baseline.
- Difficulty breakdown: schema noise, instance noise, row overlap, column overlap, and source.

### 8.4 Failures

A failed Jev or Haiku request is retried by the SDK. If it still fails, the pair is marked incomplete and excluded, and the report counts these pairs. A baseline that times out is recorded as `timeout`. Coverage is reported for each method.

## 9. Budget, time, and safety rules

- **Ledger:** `runs/ledger.jsonl` records, for each API call, the provider, tokens, and cost. Before each batch, the code checks that ledger total + batch estimate ≤ cap. If not, it stops.
- **Jev caps:** stop at $4.00 cumulative and ask the user to top up. After the top-up, the absolute stop is $9.50 (the margin covers rounding in the ledger).
- **Haiku cap:** stop at $4.50 cumulative.
- **Time:** if the baseline full run will not finish within about 36 h of its start, finish the Haiku subset + dev split pairs and show paper values for everything else.
- **Secrets:** keys are only read from `.env` by code at runtime. They are never printed, logged, or written to reports.

## 10. Report (`report/index.html`)

### 10.1 Format
- One self-contained HTML file. All JS, CSS, and data are inlined, so the page works offline at the venue.
- Light and dark themes. Large type for screen-sharing.
- Arrow keys (and `j`/`k`) jump between sections.
- Built by `scripts/50_build_report.py` from the result files only. No number is typed by hand.

### 10.2 Sections (talk order)
1. **The question:** schema matching for dataset discovery, and how it becomes a yes/no question. Includes one real Jev request and response from the logs.
2. **Setup:** datasets, the four scenarios (Fig. 2/3 of the paper), the methods table, the splits, and the tuning protocol.
3. **Accuracy:** paper-style Recall@GT box plots per scenario, with Jev modes and Haiku placed next to each method family. Filters for source and noise. A table for Magellan / WikiData / NYU next to the paper's Table IV.
4. **Cost and latency:** accuracy against cost per pair, and accuracy against wall-clock per pair, for all methods. Table V comparison (with hardware caveat). Concurrency sweep. Fan-out against singles. Sample-size sweep curve.
5. **Where Jev fails:** difficulty breakdown, win/tie/loss counts, and the worst pairs.
6. **Per-pair drill-down:** pick any pair to see the source×target heatmap of noul values with GT cells outlined. Click a cell to see the exact text Jev received and the response.
7. **Method and validity:** stability results, tie bounds, statistical tests, threats to validity (§12), and reproduction commands.

### 10.3 Citation drawer
Every number is clickable and opens its source:
- **Paper:** table or figure, page number, and a quoted label. Values read from box plots are marked "≈ read from figure".
- **Run data:** the result file, the filter that produced the number, the raw log rows, and the script path with its SHA-256.
- **Prices:** the docs URL and the date it was read (TypeSafe models page, 2026-10-30; Anthropic pricing page, date checked).

### 10.4 Slide figures
`report/figures/`: PNG (2×) and SVG, sized for 16:9, light and dark versions, for 5–6 key charts (the accuracy box plots, the cost/accuracy plot, the latency comparison, the sample-size sweep, one drill-down heatmap).

## 11. Hardware note (for the report's latency section)

- **Paper:** two 80-core, 320 GB Linux VMs (§VII-C, p. 478).
- **This run:** Intel Core i9-9880H, 16 threads, 32 GB RAM, macOS 15.7.9.
- **Jev and Haiku latency** depends on the network and on server load. The report records the time of day and the client location (this Mac).
- **The Jetson Orin Nano is not used.** The baselines use only the CPU: string edit distance, TF-IDF, Earth Mover's Distance, and integer programming. None of them has a GPU code path, and the Jetson's 6-core Arm CPU is slower than the i9.

## 12. Threats to validity (shown in the report)

1. Baselines read every row, while Jev and Haiku see N sampled values per column.
2. Today's package is a rewrite, so re-run numbers can differ from the paper's.
3. The hardware differs from the paper's. API latency depends on the network.
4. The Valentine data has been public since 2021. Whether it is in Jev's or Haiku's training data is unknown. The NYU pairs are newer and only partly reduce this risk.
5. Haiku judges all candidates together in one answer, while Jev judges each candidate on its own.
6. The prompt and the baseline configurations are tuned on the dev split. Dev pairs are excluded from the headline numbers.
7. The noise in the generated pairs is synthetic (typos, abbreviations, dropped vowels).
8. Jev is pinned to 1.14.0. Results can change with later versions.

## 13. Project layout

```
L2/
  .env                      # user-managed keys; never read by Claude
  .plans/jev-valentine-schema-matching-spec.md
  Valentine_Evaluating_Matching_Techniques_for_Dataset_Discovery.pdf
  pyproject.toml, uv.lock
  config/frozen.json        # frozen wording, N, value length, model, baseline params
  src/jevmatch/             # datasets, profiles, jev, haiku, baselines, metrics, ledger
  scripts/                  # 00_fetch_data … 50_build_report (one per phase)
  data/raw/, data/pairs.parquet
  runs/<phase>/<method>/*.jsonl, runs/ledger.jsonl
  results/*.parquet
  report/index.html, report/figures/
```

## 14. Actions for the user

1. Add `ANTHROPIC_API_KEY` to `.env` before Phase 2 (the Haiku dev check). If possible, also set a $5 spend limit on that key's workspace in the Anthropic Console.
2. Be ready to add $5 to the TypeSafe account if the Phase 1 estimate needs it.
3. Optional: check the TypeSafe rate limits in the console (<https://console.typesafe.ai/rate-limits>). Phase 1 also measures them.

## 15. Open items (decided during the run, recorded in `config/frozen.json`)

- The exact value length cap and N (from the sweep).
- The final question wording and whether to use `criteria`.
- The per-pair timeout for baselines (from pilot runtimes).
- The Haiku thinking setting.
- Whether the baseline run uses the full set or the fallback subset.

## 16. Sources

- Koutras et al., *Valentine: Evaluating Matching Techniques for Dataset Discovery*, ICDE 2021, pp. 468–479 (local PDF).
- Valentine code: <https://github.com/delftdata/valentine> (README and `experiments/`).
- Valentine datasets: <https://zenodo.org/record/5084605>.
- Jev docs, read 2026-10-30: <https://docs.typesafe.ai/primitives/noul.md>, <https://docs.typesafe.ai/models.md> (price, context limits, aliases), <https://docs.typesafe.ai/api/errors.md> (limits, 429/529), <https://docs.typesafe.ai/cookbooks/parallel_questions.md> (batching cost claim).
- Claude Haiku 5.5 pricing: Anthropic model table (cached 2026-10-06). Check against the live pricing page before publishing.

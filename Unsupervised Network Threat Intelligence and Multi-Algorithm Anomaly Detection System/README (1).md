# Unsupervised Network Threat Intelligence and Anomaly Detection System

An unsupervised machine learning project that clusters raw network traffic behavior from the **UNSW-NB15** dataset to identify anomalous / attack-like patterns — without ever using attack labels to train the model. Labels are reserved strictly for post-hoc interpretation of what the unsupervised algorithms discovered on their own.

---

## 1. Project Goal

Demonstrate a complete, honest unsupervised learning workflow using `scikit-learn`:

- Build a preprocessing pipeline that handles mixed numeric/categorical network traffic features.
- Cluster traffic using three different algorithms — **K-Means**, **Agglomerative (Hierarchical) Clustering**, and **DBSCAN** — so their strengths/weaknesses can be compared.
- Only *after* clustering is complete, bring in the `label` (Normal/Attack) and `attack_cat` (attack type) columns to evaluate and interpret what the clusters found — never to shape them.

This mirrors a realistic security-analytics scenario: you rarely have labeled attack data in real time, so the goal is to see how well purely behavioral clustering surfaces anomalous traffic on its own.

---

## 2. Dataset

**UNSW-NB15** (Kaggle), provided as a ZIP containing:

- `UNSW_NB15_training-set.parquet`
- `UNSW_NB15_testing-set.parquet`

Both files are loaded and concatenated into a single working dataset (the train/test split isn't relevant here since there's no supervised training step).

| | |
|---|---|
| Combined shape | 257,673 rows × 36 columns |
| Numeric columns | ~30 (traffic timing, byte counts, rates, jitter, TCP flags, connection counters) |
| Categorical columns | `proto`, `service`, `state` |
| Label columns (held out) | `label` (0 = Normal, 1 = Attack), `attack_cat` (10 categories: Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms) |
| Missing values | None detected |
| Duplicate rows | 112,451 (43.6% of combined data) |

---

## 3. Methodology

### Step 1 — Dataset Loading & Initial Inspection
Extracts the ZIP, auto-detects CSV/Parquet files, loads and concatenates them, and profiles the raw data: shape, dtypes, missing values, duplicates, and summary statistics.

### Step 2 — Preprocessing Pipeline (label-free)
- `label` and `attack_cat` are split off into a separate `df_labels` DataFrame and excluded from every training step.
- Identifier-like columns are dropped.
- Numeric columns → median imputation → scaling.
- Categorical columns (`proto`, `service`, `state`) → most-frequent imputation → one-hot encoding.
- Combined via a single `ColumnTransformer` into one feature matrix.

### Step 3 — Clustering
All three algorithms are fit **only** on the processed feature matrix (no labels):
- **K-Means** — model order (`k`) selected via elbow (inertia) and silhouette score sweep across k=2–10.
- **Hierarchical (Agglomerative, Ward linkage)** — fit with the same `k` chosen for K-Means for a fair comparison; a dendrogram is plotted for structural inspection.
- **DBSCAN** — density parameter `eps` estimated from a k-distance graph; naturally discovers its own number of clusters and flags low-density points as noise (`-1`).

A 20,000-row random subsample is used for clustering to keep Hierarchical/DBSCAN computationally tractable.

### Step 4 — Cluster Interpretation (labels used for the first time, read-only)
Cross-tabulates each algorithm's cluster assignments against `label` and `attack_cat` to see what kind of traffic ended up in each cluster, including which clusters skew toward attack traffic and what DBSCAN's noise points look like compositionally.

### Step 5 — Dimensionality Reduction & Visualization
PCA and t-SNE project the (label-free) feature matrix into 2D for visual inspection, then the same 2D coordinates are re-used to color plots by cluster assignment and, separately, by ground-truth label — for visual comparison only.

### Step 6 — Formal Evaluation
External validation metrics (Adjusted Rand Index, Normalized Mutual Information, Homogeneity, Completeness, V-Measure, Fowlkes-Mallows) compare each algorithm's clusters against `label` and `attack_cat`. Internal validation (silhouette score, label-free) is reported alongside for a complete scorecard. This step is reporting-only — it happens after all clustering decisions are already final.

### Enhancement 1 — Cluster Behavioral Profiling
For each K-Means cluster, computes size, the numerical features that deviate most from the overall dataset average, and the most common categorical values, then generates a cautious, hedged interpretation (e.g. "pattern potentially consistent with high-frequency, short-duration connection behavior"). Deliberately avoids definitive claims like "this is a port scan" — output is descriptive, not diagnostic.

### Enhancement 2 — Post-Hoc Cluster Explainability
Trains a Random Forest **only** to predict the already-discovered K-Means cluster labels from the processed features — explicitly a post-hoc explainability tool, not part of the clustering model and not an attack classifier. Reports the top 10 features that most separate the clusters, with a feature-importance bar chart.

### Enhancement 3 — Ensemble Threat Risk Scoring (Experimental)
Combines three label-free clustering signals into a single 0–100 score: DBSCAN noise status (40%), distance from the assigned K-Means centroid (35%), and K-Means cluster rarity (25%). Records are bucketed into Low/Medium/High/Critical Risk. Explicitly documented as an **experimental, unvalidated prioritization score** for exploratory analysis — not a production security control.

### Enhancement 4 — Automated Threat Intelligence Report
Prints a consolidated, report-style summary pulling together total records analyzed, cluster counts per algorithm, DBSCAN anomaly rate, Step 6's evaluation metrics, the risk-score distribution from Enhancement 3, and the top features from Enhancement 2 — formatted for inclusion in a project write-up.

### Step 11 — Post-Hoc Threat Score Validation
Checks whether the experimental risk score from Enhancement 3 actually prioritizes attack traffic, using `label` strictly for validation — never to modify the score. For each risk level (Low/Medium/High/Critical), computes total records, normal count, attack count, and attack/normal percentages, visualizes attack percentage by risk level, and runs an automated (not forced) trend check across Low → Medium → High → Critical. If the trend isn't consistently increasing, the notebook reports that honestly rather than asserting success.

### Step 12 — Attack Category Enrichment Analysis
Uses `attack_cat` strictly for post-hoc interpretation. For each risk level, computes the attack-category distribution, top 5 categories, and percentage contribution (grouping minor categories as "Other" for chart readability), then does a focused deep-dive on High + Critical Risk records specifically. Generates a cautious interpretation of category over-representation only where the numbers actually support it, and produces a final summary table: Risk Level, Total Records, Normal %, Attack %, Most Common Attack Category.

---

## 4. How to Run

1. Download the UNSW-NB15 dataset ZIP from Kaggle.
2. Open the notebook and update the `zip_path` variable in Step 1 to point to your ZIP file.
3. Run cells top to bottom. Each step prints its own diagnostics (shapes, counts, scores) so you can sanity-check progress before moving to the next step.

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`

---

## 5. Results Summary (current run)

### Clustering overview

| Algorithm | Clusters found | Silhouette (internal) |
|---|---|---|
| K-Means | 2 (best by silhouette sweep) | 0.494 |
| Hierarchical | 2 | 0.487 |
| DBSCAN | 226 (+ noise) | 0.525 |

### Alignment with ground truth (external validation)

| Algorithm | ARI vs label | NMI vs label | ARI vs attack_cat | NMI vs attack_cat |
|---|---|---|---|---|
| K-Means | 0.108 | 0.076 | 0.130 | 0.153 |
| Hierarchical | 0.126 | 0.097 | 0.152 | 0.167 |
| DBSCAN | 0.092 | 0.172 | 0.367 | 0.374 |

DBSCAN's noise points (1,447 of 20,000, ~7.2%) contained 58.5% attack traffic, close to the sample's overall 64.1% attack rate — i.e. in the current run, noise wasn't yet a strongly concentrated attack signal.

### ⚠️ Known limitation in the current run

The preprocessing step's numeric-column filter (`select_dtypes(include=["int64", "float64"])`) does not match this dataset's actual dtypes (`int8`/`int16`/`int32`/`float32`), so **only 2 of ~32 numeric columns** (`stcpb`, `dtcpb` — TCP sequence numbers) were included as numeric features; the ~30 genuinely behavioral numeric columns (byte counts, rates, timing, jitter) were silently excluded. The results above are therefore driven almost entirely by one-hot-encoded `proto`/`service`/`state`, which explains DBSCAN's high fragmentation (226 clusters) and the non-monotonic K-Means silhouette curve.

**Fix:** use `select_dtypes(include=np.number)` instead, and drop `stcpb`/`dtcpb` as near-random identifiers rather than treating them as the primary numeric signal. Re-running Steps 2–6 after this fix is expected to materially change cluster quality and label alignment, since the algorithms will finally see real traffic-behavior features.

---

## 6. Design Decisions & Rationale

- **Labels are never used for training or clustering** — held out from Step 2 onward and only reintroduced in Step 4 for interpretation, Step 6 for evaluation, and Steps 11–12 for risk-score validation, matching the "unsupervised" framing of the project.
- **20,000-row subsample for clustering** — Hierarchical clustering and DBSCAN don't scale well to 257K+ rows on standard hardware; the subsample keeps the notebook runnable while preserving enough data for meaningful cluster structure.
- **Same `k` used for K-Means and Hierarchical** — chosen via K-Means' silhouette sweep, applied to both so their outputs are directly comparable.
- **t-SNE run on a further 5,000-point sub-sample** of the clustering sample — t-SNE's complexity scales roughly quadratically, so this keeps runtime reasonable; index alignment (`tsne_idx`) is tracked carefully so cluster/label colors line up correctly with the smaller t-SNE coordinate set.
- **Threat risk score is explicitly experimental** — a transparent, fixed-weight combination of DBSCAN noise status, K-Means centroid distance, and cluster rarity, documented as unvalidated. Steps 11–12 check it against ground truth after the fact but never feed that check back into the score itself.

---

## 7. Next Steps / Possible Extensions

- Fix the numeric-dtype selection bug (see Section 5) and re-run the full pipeline.
- Remove the 112,451 duplicate rows before clustering to prevent a handful of repeated traffic patterns from dominating cluster density.
- Apply `RobustScaler` or a `log1p` transform to heavy-tailed features (e.g. `dbytes`, `sload`) before `StandardScaler`, since a few extreme values currently have outsized influence on Euclidean distance.
- Re-tune DBSCAN's `eps` via a fresh k-distance graph after the feature set changes.
- Try dimensionality reduction (e.g. PCA to 20–30 components) before clustering, rather than clustering directly on the full one-hot-expanded feature space.
- Enhancements 1–4 and Steps 11–12 all build directly on `X_cluster`/`kmeans_labels`/`risk_score`, so re-run them after fixing the Section 5 dtype bug — otherwise they'll faithfully explain and validate clusters still driven mostly by `proto`/`service`/`state` rather than real traffic behavior.

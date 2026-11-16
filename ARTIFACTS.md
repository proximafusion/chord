# Artifacts

Prepared artifacts and their provenance. They support evaluation of saved results;
they do not include every training corpus needed to retrain the paper's models.
The Hugging Face dataset is private and requires an authenticated account with read
access. Revision `ad7d496e918cf193e79076418426ed314a07d1de` contains the boundary corpus,
corrected campaign table, generated front, feasible boundary JSON, and an audit report.

```bash
# The default; private-dataset access is required.
export CHORD_ARTIFACT_BASE=hf://datasets/proxima-fusion/chord-paper
```

Any location holding a copy works, including a local directory. The scheme selects the
backend, so `gs://` needs `pip install 'chord[gcs]'` and `hf://` needs
`chord[huggingface]`.

| file | size | location | what it is |
|---|---|---|---|
| `corpus.parquet` | 80 MB | Hugging Face and GCS | the fixed-aspect-ratio design corpus |
| `surrogate_corpus.parquet` | 374 MB | GCS only | a subset of the surrogate's training and validation data |
| `surrogate.pt` | 2.3 MB | GCS only | the prepared surrogate checkpoint |
| `campaign_front.parquet` | 324 kB | Hugging Face | corrected campaign roles for front comparisons |
| `generated_front.parquet` | 2.7 MB | Hugging Face and GCS | generated designs and high-fidelity evaluations |
| `generated_boundaries.json` | 2.3 MB | Hugging Face and GCS | the 1,120 benchmark-feasible generated boundaries |

The two surrogate artifacts have not been uploaded to Hugging Face. A local copy
containing both files can be used with `CHORD_ARTIFACT_BASE=/path/to/artifacts`.
Their original storage locations are retained in the authors' provenance records.
The GCS campaign table still has stale roles; use the corrected Hugging Face table
for the paper's five-anchor comparison.

## `corpus.parquet`

100,000 plasma boundaries with their simulated metrics, split 80/10/10 into train,
validation and test. Roughly 55% lie in the planar-axis basin and 45% in the torsion
basin. The interpolation experiments pair designs from these two geometry families;
the sampled separation does not establish topological disconnection.

Each row carries `boundary_coefficients`, a 90-element vector of Fourier coefficients in
the layout `chord.features` defines, alongside 14 metrics. Metric names and units follow
constellaration, so the rotational transform is per field period and the magnetic
gradient scale length is scaled by the field-period count.

```python
from chord import corpus

train = corpus.load_corpus().subset("train")
```

The split is the one the released checkpoints were fitted against rather than a fresh
one, so evaluating a released model on the test split does not score it on its own
training data.

Every boundary here has **aspect ratio 7.0**. The corpus was built to study feasibility
at a fixed aspect ratio. Rescaling it with `chord.experiments.frontfill.augment`
can construct a new training experiment, but does not reconstruct the paper's
multi-aspect-ratio anchor-campaign training corpus.

## `surrogate_corpus.parquet`

500,000 boundaries sampled from the broad-scan surrogate dataset, keeping its
train and validation proportions. Boundaries the simulator failed on are dropped: they
are about one in ten thousand, and with no convergence head to train there is nothing to
do with a row whose metrics are all missing.

## `surrogate.pt`

Predicts 15 metrics from a standardized boundary. The checkpoint carries its own metric
normalization **and its own feature standardization**, and the loader refuses one that
does not.

That second part is load-bearing. This surrogate was fitted on a superset of the corpus
above, and its per-column feature statistics differ. Standardizing boundaries against
the corpus instead and feeding them to these weights drops agreement with the simulator
on feasibility from 93% to 25%, with no error raised anywhere.

```python
from chord import artifacts

predictor = artifacts.released_surrogate()
metrics = predictor.predict_boundaries(train.boundaries[:64])
```

`predict_boundaries` goes from raw coefficients to metrics using the transform the
weights were actually fitted against, which is the call to reach for.

## `campaign_front.parquet`

3,572 designs from optimization campaigns on the MHD-stable quasi-isodynamic benchmark,
431 of them feasible. This is what front-filling is scored against rather than something
it learns from.

`role` records what each campaign is for: `anchor` campaigns are the only ones a
front-filling model may see, `held_out` ones are withheld and keep their fronts as
ground truth, `excluded` ones are left out of the study, and `outside` covers the rest
and the external baselines.

Roles come from each campaign's **target** aspect ratio, not the value it achieved. A
campaign aiming at 8.00 that lands at 8.04 is still the 8.00 campaign.

```python
from chord.experiments.frontfill import campaign

report = campaign.score(campaign.load())
```

The scorer distinguishes the study reference from the broader benchmark:

| scorer key | comparison | hypervolume |
|---|---|---:|
| `anchors_only` | five retained campaigns | 174.5553 |
| `study_campaign` | five anchors plus six held-out campaigns | 176.0221 |
| `full_campaign` | all feasible campaign rows, including excluded campaigns | 178.4480 |
| `anchors_plus_generated` | anchors plus the 1,120 benchmark-feasible generated designs | 176.3135 |

The broader reference agrees with the published benchmark figure of 178.4; it must
not be labeled the eleven-campaign study reference. `recovered` uses that broader
denominator; `recovered_study` uses the eleven-campaign denominator. These fractions
can exceed one when generation improves on the chosen reference.

Hugging Face revision `ad7d496` corrects 160 rows from the A=10.00 campaign:
they are excluded, not grouped with the nearby A=10.04 anchor. Their metrics are
unchanged. The corrected anchors have 16 distinct non-dominated objective pairs;
adding the generated designs gives 51: 37 generated points and 14 surviving anchors.
The scorer identifies generated points by provenance, not by their position relative
to rounded anchor targets. It also reports the largest adjacent aspect-ratio gap
along the front, which decreases from 1.25 to 0.5076. The exact point count depends
on numerical resolution: rounding both objectives to six decimal places gives
5 anchor and 42 combined points while preserving the reported hypervolumes.
The source-builder
fix on `scadena-ca-batch-a` is still needed before regenerating the GCS table.

## `generated_front.parquet`

1,200 designs CHORD generated to fill the gaps between anchor campaigns, 300 per gap,
each carried as both a coefficient vector and a boundary in the benchmark's own JSON
format. These are selected from 8,361 low-fidelity-feasible candidates. Of the 1,200
high-fidelity evaluations, 1,120 satisfy the benchmark's normalized-violation tolerance
of 0.01; 977 satisfy the nominal bounds strictly. These rates describe the screened
subset, not all generated proposals.

The boundary coefficients and JSON representations allow these generated designs
to be re-evaluated with a specified simulator version and fidelity.

Applying the nominal constraints without the benchmark tolerance to both campaign
and generated rows leaves 85 feasible anchors and 977 feasible generated designs.
Hypervolume increases from 167.7518 to 172.2466 and the largest aspect-ratio gap
falls from 1.25 to 0.3406. This is a separate sensitivity calculation; do not combine
strict generated rows with tolerance-filtered anchors.

## Reproduction levels and version requirements

The `stellarator` extra pins `constellaration==0.2.6`, preserving the version
identified by the `scadena-chord-pin-simulator` audit. Version 0.3 changes the flux-compression
definition and therefore the benchmark's feasible set. Saved-metric rescoring does
not require installing or running VMEC.

Pin the dataset revision when downloading, then point the code to that local copy:

```bash
hf download proxima-fusion/chord-paper --repo-type dataset --revision ad7d496e918cf193e79076418426ed314a07d1de --local-dir artifacts/chord
export CHORD_ARTIFACT_BASE=artifacts/chord
```

| task | available inputs | limitation |
|---|---|---|
| Corpus analysis and front rescoring | pinned private Hugging Face package | requires repository access |
| Re-evaluating released boundaries | package plus the pinned simulator | requires new physics evaluations |
| Rescoring the original fixed-AR interpolation sweep | candidates and metric shards in internal GCS | historical tables use eight interior points; the release evaluator includes all nine |
| Retraining the original front models | per-gap data, models, and configs in internal GCS | not part of the Hugging Face package |

The original main interpolation sweep has been rescored on the symmetric nine-point
grid as an appendix sensitivity analysis. The historical eight-point comparisons
are retained consistently for slerp, flow matching, and target tracking; their raw
rebuttal evaluations were not recovered in the current audit. A fresh nine-point
run should be compared to the nine-point appendix table, not the historical table.
Fresh three-seed flow-matching and cycle-consistency-proxy runs have since completed
and are archived separately. They support comparisons under their recorded protocol;
they are not replacements for missing historical raw files, and the cycle proxy is
not a reproduction of Notin's Bayesian estimator.

Per-gap reproduction artifacts have the following original paths relative to
the archived artifact root. The prepared release layout is documented in
[REPRODUCING.md](REPRODUCING.md):

- `arc_datasets/arc_cvae_v5_g{1,2,3,4}.npz`: saved generative training/validation data.
- `arc_datasets/arc_surrogate_v5_g{1,2,3,4}.npz`: saved surrogate data and normalization.
- `arc_chord_v5_g{1,2,3,4}/config.json`: architecture and training configuration.
- `arc_chord_v5_g{1,2,3,4}/models/arc_cvae_z16_chord{0,0.01}.pt`: model weights.
- `arc_surrogate_v5_g{1,2,3,4}/models/surrogate.pt`: per-gap surrogate weights.
- `arc_v5_eval/arc_v5_front_g{1,2,3,4}_lambda{0,0p01}.json`: saved sweeps and
  high-fidelity checks of candidates selected on low-fidelity Pareto fronts.

The archived per-gap high-fidelity columns count feasible survivors of low-fidelity
Pareto screening, not a recomputed high-fidelity Pareto front. Across all four gaps,
both regularizer arms used 11,406 low-fidelity and 109 high-fidelity evaluations.
Their saved feasible objective pairs, combined with strict anchors at the archives’
four-decimal precision, give hypervolume 172.26 without regularization and 172.79
with it. These are separate from the 1,200-candidate stratified benchmark evaluation.

The braces above describe families of object names, not artifact-loader syntax.
Exact rebuilding of the saved datasets additionally requires source queues,
augmentation provenance, and their metric shards. Internal availability is not a
completed public release; dataset visibility remains private.

## Prepared front reproduction package

The expanded release candidate stages the four generative datasets, four surrogate
datasets, all eight decoder checkpoints, four surrogate checkpoints, and original
configurations from pinned GCS object generations. Its standalone loaders preserve
the saved splits, conditioning order, and separate feature statistics. The numerical
arrays are unchanged; string metadata uses a pickle-free representation.
See [REPRODUCING.md](REPRODUCING.md) for the layout, inference commands and parity
checks. This package is separate from the currently pinned Hugging Face revision.

## What is not in the pinned Hugging Face revision

The paper's per-gap training corpus is **not** included. Those models were trained
on their two bracketing anchor campaigns plus simulator-evaluated rescalings of
boundaries from those campaigns. Held-out campaigns and their rescaled descendants
were excluded by provenance. The fixed-AR=7 `corpus.parquet` is a different source
dataset and cannot reconstruct that training pool.

The released augmentation code rescales boundaries and re-evaluates their metrics.
Its entry point now requires an explicit anchor-campaign corpus and checks campaign
provenance before simulation. The prepared package bypasses this expensive build
step for saved-model inference. Exact retraining still needs the archived trainer's
sampling and optimization protocol, not just matching input arrays.

Regenerating it is not free. Every rescaled boundary has to go back through the
simulator, because rescaling changes the physics and a source's metrics describe a
different configuration. At roughly 50 vCPU-seconds per boundary, a few hundred thousand
rows is a few thousand vCPU-hours.

Accuracy of the rescale, measured against the simulator: aiming six published boundaries
at targets from 8 to 12 lands within 0.03 on average and 0.13 at worst, with the error
growing with distance from the source's own aspect ratio. That approximation costs
nothing, because what a boundary ends up at is read back from the simulator rather than
assumed.

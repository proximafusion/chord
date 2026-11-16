# Artifacts

Data and trained models are hosted on
[Hugging Face](https://huggingface.co/datasets/proxima-fusion/chord-paper).
The code pins revision `28f84539fddd171d950d5f7d4a8878dcc649ad15`. Loaders fetch the artifacts needed for each
workflow; the full optimization history is not required for saved-model inference.
`CHORD_ARTIFACT_BASE` can select another revision or a downloaded local directory.

| Artifact | Contents |
|---|---|
| `corpus.parquet` | 100,000 feasible fixed-aspect-ratio boundaries and original train/validation/test labels |
| `surrogate_corpus.parquet` | 500,000 sampled boundaries for compact surrogate-training studies |
| `surrogate.pt` | Original fixed-AR metric surrogate, with its feature and metric normalization |
| `fixed_ar/` | 48 sweep models plus one alternate checkpoint, a model catalog, and original transforms |
| `frontfill/g1/` through `g4/` | Eight decoders, four surrogates, original configurations, generator inputs, and surrogate normalization |
| `campaign_front.parquet` | Corrected campaign roles and front-comparison metrics |
| `generated_front.parquet` | All 1,200 screened front candidates and high-fidelity evaluations |
| `generated_boundaries.json` | The 1,120 benchmark-feasible generated boundaries |
| `release_manifest.json` | Sizes and SHA-256 hashes for the added data and model files |

The existing corpus and front-evaluation files are unchanged from revision
`ad7d496e918cf193e79076418426ed314a07d1de`. The new release adds the surrogate subset,
checkpoints, inference inputs and documentation. `audit.json` describes the original
data checks; `release_inference_checks.json` and `surrogate_subset_audit.json` record
the added checks. Install `chord[huggingface]` to download artifacts, or use
a local copy through `CHORD_ARTIFACT_BASE`.

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

500,000 boundaries sampled within the original surrogate train and validation
pools: 450,000 training and 50,000 validation rows. The file contains 90 raw
Fourier coefficients and 11 physical metrics per row. All entries are finite.
There are 499,995 distinct boundary vectors and five repeated rows.

This is the existing random subset, not a diversity-selected or near-duplicate-
filtered dataset. Rows missing any original target metric were excluded during
export. It supports compact training studies, but retraining on it has not been
shown to reproduce the historical 15-output surrogate. Its full 5,089,970-row
training/validation archive is available upon request.

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
When rebuilding the campaign table, preserve these corrected roles rather
than reassigning campaigns from their achieved aspect ratios.

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
used for the audited benchmark feasibility calculation. Version 0.3 changes the flux-compression
definition and therefore the benchmark's feasible set. Saved-metric rescoring does
not require installing or running VMEC.

Pin the dataset revision when downloading, then point the code to that local copy:

```bash
hf download proxima-fusion/chord-paper --repo-type dataset --revision 28f84539fddd171d950d5f7d4a8878dcc649ad15 --local-dir artifacts/chord
export CHORD_ARTIFACT_BASE=artifacts/chord
```

| task | available inputs | limitation |
|---|---|---|
| Corpus analysis and front rescoring | pinned Hugging Face package | no new simulator evaluation |
| Re-evaluating released boundaries | package plus the pinned simulator | requires new physics evaluations |
| Rescoring the original fixed-AR interpolation sweep | historical candidate and evaluation files, available upon request | historical tables use eight interior points; the release evaluator includes all nine |
| Saved-model inference | fixed-AR and per-gap checkpoints on Hugging Face | original normalization included |
| Retraining the original front models | prepared generator inputs and configs on Hugging Face; complete surrogate inputs upon request | archived training protocol still needs a validated standalone port |

The original main interpolation sweep has been rescored on the symmetric nine-point
grid as an appendix sensitivity analysis. The historical eight-point comparisons
are retained consistently for slerp, flow matching, and target tracking; their raw
rebuttal evaluations were not recovered in the current audit. A fresh nine-point
run should be compared to the nine-point appendix table, not the historical table.
Fresh three-seed flow-matching and cycle-consistency-proxy runs have since completed
and are archived separately. They support comparisons under their recorded protocol;
they are not replacements for missing historical raw files, and the cycle proxy is
not a reproduction of Notin's Bayesian estimator.

## Saved-model inference

`training.load_archived_model()` loads a checkpoint from the catalog in
`fixed_ar/models.json`. The catalog includes VAE, cVAE, unconditional CHORD and
CHORD at latent dimensions 4, 8, 12 and 16 for three training seeds. The additional
seed-zero z=8 CHORD checkpoint is named `chord_alternate.pt` and retains its own
source-version label. Each bare state dictionary is unchanged from the archive.
The shared metadata stores original decoder transforms; conditioning names and
units match the released corpus. The decoder's feature normalization must not be
replaced by the surrogate's normalization.

For the four front gaps, the released prepared generator arrays retain their
original splits. Small `surrogate_metadata.npz` files provide the surrogate's
normalization without requiring the full per-gap surrogate-training arrays.
All eight decoder checkpoints and four surrogate checkpoints have been checked
with the standalone loaders. See [REPRODUCING.md](REPRODUCING.md) for commands.

## Full historical data available upon request

Contact [Santiago A. Cadena](mailto:scadena@proximafusion.com) for the complete
archives used in the historical experiments:

- The 1,010,066-design feasible pool and its original grouped splits.
- The separate 5,089,970-row surrogate dataset, including all 15 target metrics,
  missing-value masks, original 90/10 split and preprocessing statistics.
- The complete per-gap surrogate training arrays and source-campaign provenance.

These archives preserve the original optimization histories, including similar
late-campaign boundaries. The compact release avoids requiring their download
for inference. The million-design feasible pool is not a substitute for the
full surrogate dataset, which also contains constraint-violating designs.

Reproducing historical training additionally requires the archived optimizer,
sampling and validation protocol. A model trained using the general-purpose
examples or the compact subset is a new training run, not an established
reproduction of the paper's checkpoints. No claim of exact retraining parity is
made by the inference checks.

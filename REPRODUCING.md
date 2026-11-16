# Reproducing CHORD

The fixed-AR interpolation experiment and the campaign-front experiment use
different training data, conditioning vectors, and constraints. A successful
demonstration run is not evidence that it reproduces a paper table.

## Verified scope

| Workflow | Status | Evidence or requirement |
|---|---|---|
| Rescore the released front | Available | Pinned campaign and generated-front tables in `ARTIFACTS.md` |
| Decode archived per-gap checkpoints | Checked against original inference | All eight decoders, four surrogates, plain interpolation and two self-consistency variants |
| Re-evaluate decoded boundaries | Requires physics evaluation | ConStellaration 0.2.6 and the recorded fidelity settings |
| Train a new model with the examples | Available | Examples exercise the current implementation |
| Retrain the reported front models | Not yet validated in the standalone package | Requires the historical pair sampler, optimizer/scheduler, and validation protocol |

The inference comparison used 24 candidates for each of 24 combinations of gap,
model and inference setting: 576 boundaries in total. All decoded coefficients
matched exactly on CPU with PyTorch 2.5.1 and NumPy 2.2.6. The independent metric
head checks also matched exactly. This checks the port on fixed inputs, not
retraining reproducibility or renewed simulator feasibility.

## Prepared inputs

The expanded release candidate has this layout; it is staged separately from the
Hugging Face revision documented in `ARTIFACTS.md`:

```text
manifest.json
inference_parity.json
frontfill/g1/                 # also g2, g3, g4
  dataset.npz
  surrogate_dataset.npz
  surrogate_metadata.npz
  config.json
  chord.pt
  cvae.pt
  surrogate.pt
```

The manifest records original GCS generations and checksums, plus the checksums
of the release files. Numeric arrays are unchanged. Object arrays containing
names and a boundary template are converted to Unicode so loading does not need
NumPy pickle support. Weights are loaded with `weights_only=True`.

The generator NPZ retains its training, validation and held-out test splits.
Endpoints for inference come only from the training split. The conditioning order
is read from the artifact: iota per field period, log10 QI, mirror ratio, flux
compression, vacuum well, elongation, gradient scale length and aspect ratio.
The generator and surrogate have separate feature standardization; inference
explicitly converts between them before requesting metric predictions.

```bash
export CHORD_ARTIFACT_BASE=/path/to/prepared-release
python -m chord.experiments.frontfill.archive --gap g1 --model chord --out g1.npz
python -m chord.experiments.frontfill.archive --gap g1 --model cvae --out g1_cvae.npz
python -m chord.experiments.frontfill.archive --gap g1 --iterations 15 --out g1_sc.npz
```

Here alpha is the weight on the **low** anchor, matching the archived protocol.
Default sampling uses 19 weights from 0.05 to 0.95 and 60 independently sampled
pairs per weight. Outputs include raw coefficients, blend weights, endpoint
indices and the command settings. They carry no feasibility labels.

The small `surrogate_metadata.npz` is sufficient for inference; the larger
surrogate dataset is retained for training and split audits. The source row and
campaign lineage used to build the NPZs is not reconstructed by these loaders.
Rebuilding those NPZs from original simulation shards is a separate workflow.

## Feasibility and simulator versions

Use ConStellaration **0.2.6** for comparisons with the paper. The 0.3.0 release
changed the flux-compression calculation and therefore the feasible set. The
existing simulator-pin audit recovered the stored hypervolume and all 57
feasible designs under 0.2.6; 0.3.0 rejected nine of the same boundaries.

Benchmark front feasibility includes its 1% normalized-violation tolerance.
For the screened 1,200-candidate release, this gives 1,120 feasible designs
(93.33%). Applying the nominal bounds strictly gives 977 (81.42%). Compare
campaigns and generated designs using the same rule. Neither rule should be
substituted for the looser constraints in the fixed-AR interpolation experiment.

## Remaining retraining work

The current general-purpose training loop is not the archived front trainer.
The latter used Adam, learning-rate warmup followed by a scheduler, gradient
clipping, dropped incomplete training batches, and a regularizer combining generic
pairs with extra weight on pairs separated in aspect ratio. Validation also
consumed random draws. These choices must be ported and compared on fixed seeded
minibatches before claiming retraining parity. Reusing the current `fill` command
and merely matching the epoch count would not establish it.

The prepared data and checkpoints remove the need to regenerate costly simulation
campaigns for this work. Full training and new large simulator studies should run
on remote compute, not as release smoke tests.

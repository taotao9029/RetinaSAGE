# RetinaSAGE: Static-anchored quality-guided few-frame learning for retinal video classification

RetinaSAGE is a research codebase for binary classification of retinal videos using a frozen image encoder and a small, quality-aware set of frames. The current experimental line treats a video as a sequence of candidate observations: a static image representation provides the baseline, while quality-guided frame selection and low-capacity aggregation test whether a few additional observations improve video-level predictions.

> **Research status.** The repository contains experimental code and results, not a clinically validated diagnostic system. The reported evaluation is an internal, single-cohort, five-fold nested cross-validation study on 128 videos. Independent external validation is not yet available.

## Abstract

Retinal video classification must balance temporal coverage, image quality, computational cost, and limited labeled data. RetinaSAGE studies a static-anchored strategy: extract frozen DINOv3 features from sampled frames, use a label-free image-quality score to select a small temporally distributed subset, and aggregate the selected features with a low-capacity gated-attention multiple-instance learning (MIL) classifier. The resulting MIL predictions can be combined with a single-frame baseline through calibration fitted using training-fold data only. Frame-mean controls distinguish gains from quality-guided selection and gains from simply averaging multiple frames.

In the current 128-video evaluation, the calibrated constrained MIL-4 configuration has the strongest reported threshold-dependent metrics and Brier score among the five compared methods; uniform four-frame feature averaging has the strongest AUC and average precision. These results are preliminary and should be interpreted alongside the small, imbalanced cohort, the single cross-validation repeat, and the lack of external validation.

## Method at a glance

```text
Video
  └─ sample frames at the configured temporal rate (maximum 128)
       └─ frozen DINOv3 ViT-S/16 feature extraction
            ├─ global image embedding
            └─ Frangi-weighted regional vessel features
                 └─ per-frame feature vectors
                      ├─ B1: one-frame feature → logistic regression
                      ├─ Quality-4: select four quality-ranked, time-covered frames
                      │    ├─ feature mean → low-capacity classifier
                      │    └─ gated-attention MIL → video probability
                      └─ Uniform-4: uniformly spaced four frames → feature mean

B1 probability + MIL-4 probability
  └─ inner-fold Platt calibration and constrained fusion
       └─ outer-fold video-level prediction
```

The acronym **SAGE** denotes the research design—static anchoring, quality guidance, and efficient few-frame aggregation. It should not be read as a claim that every branch or proposed extension in the repository has been externally validated.

## Current experimental results

The table below is the fold-wise mean over the five outer folds for the 128-video experiment in `runs/B1_B5`. It is not a pooled out-of-fold metric. The same outer split manifest is used for the compared methods. Values are rounded to four decimal places.

| Method | ROC-AUC ↑ | Average precision ↑ | Accuracy ↑ | Balanced accuracy ↑ | Brier score ↓ |
|---|---:|---:|---:|---:|---:|
| B1: frozen DINOv3 single-frame logistic regression | 0.9733 | 0.9864 | 0.9846 | 0.9690 | 0.0319 |
| B1 + same final calibration (λ = 0 control) | 0.9733 | 0.9864 | 0.9769 | 0.9548 | 0.0257 |
| B1 + constrained MIL-4 + final calibration | 0.9783 | 0.9907 | **0.9923** | **0.9833** | **0.0176** |
| B1 + quality-selected 4-frame feature mean + same calibration | 0.9800 | 0.9918 | 0.9692 | 0.9683 | 0.0213 |
| B1 + uniformly selected 4-frame feature mean + same calibration | **0.9833** | **0.9937** | 0.9769 | 0.9733 | 0.0219 |

**Interpretation.** In this evaluation, calibrated constrained MIL-4 leads on accuracy, balanced accuracy, and Brier score, while uniform-4 mean leads on ROC-AUC and average precision. The quality-4 mean does not outperform uniform-4 mean on these aggregate metrics, so these results do not establish that quality ranking itself causes a benefit. The calibration-only control changes threshold-dependent metrics and Brier score, but leaves ranking metrics unchanged, as expected for a monotonic calibration transform. The table is descriptive; the repository does not claim statistical significance or external generalization from this single five-fold run.

### Residual and PCA experiments in `runs/B1_B5`

The same experiment root also contains earlier residual/PCA branches. They are reported separately from the five-method RetinaSAGE comparison above: each uses five outer folds on the 128-animal cohort, but the model inputs and training setups differ. The values below are the mean of the five outer-fold summaries, not pooled predictions.

| Experiment | Method | ROC-AUC ↑ | Accuracy ↑ | Balanced accuracy ↑ | Main finding |
|---|---|---:|---:|---:|---|
| B1 exact-anchor dynamic residual (`b1_exact_residual`) | Keep the B1 first-frame logistic-regression logit as the anchor; summarize PCA-compressed changes in global DINOv3 features relative to the first frame; train a small bounded residual correction. The residual scale includes `α = 0` as an exact B1 fallback, selected using inner OOF predictions. | 0.9733 | 0.9846 | 0.9690 | Matches the B1 aggregate metrics; it did not demonstrate a reliable improvement over the static anchor. |
| Static-anchor temporal residual (`s1_s3/static_residual`) | Fuse a first-frame logistic-regression logit with a bounded correction from continuous, consecutive standardized frame-feature differences; no GAT is used. | 0.9682 | 0.9455 | 0.9411 | The dynamic correction did not surpass the B1 reference in this run. |
| B1 + PCA-local fusion (`b1_plus_pca_local`) | Preserve B1's global CLS feature and concatenate it with PCA-compressed first-frame Frangi-weighted local-region mean/max features. Fit scaling/PCA within each training fold and select the local-feature weight by inner OOF results. | 0.9733 | 0.9535 | 0.9461 | AUC is similar to B1, but accuracy and balanced accuracy are lower; the local PCA branch did not show a clear added value. |

These experiments test different hypotheses: whether temporal changes can add a small correction to a static prediction, and whether compressed local vessel features add information beyond the global representation. Their results do not support a general claim that residual learning or PCA improves classification. In particular, the exact-anchor residual's selected `α = 0` fallback allows the model to retain B1 when the inner-fold evidence does not favor a correction; matching B1 should be interpreted as a safe fallback, not as a gain. The PCA-local experiment uses a first-frame local summary, so it does not test few-frame aggregation.

The code entry points are `train_b1_exact_residual.py` and `train_b1_pca_local.py`; the related sequence-residual experiment is implemented in the `s1_s3` experiment runner. Run manifests and fold-level predictions are under the corresponding `runs/B1_B5` subdirectories. They are ignored by the repository's `.gitignore`; only publish them if the data-governance rules permit it.

The source artifacts are `runs/B1_B5/五种实验最终对比表.md`, `runs/B1_B5/five_method_comparison.csv`, and the corresponding outer-fold metrics and manifest files. Results should be regenerated from fold-level predictions when configurations or source artifacts change.

## Method details

### Frozen frame representation

The current main comparison uses frozen **DINOv3 ViT-S/16** frame features. No backbone fine-tuning is performed in the reported B1/MIL-4 comparison. The feature pipeline combines a global image embedding with regional features weighted using a Frangi vesselness response. Features are cached and indexed through a feature-extraction manifest to avoid repeated image encoding.

### Label-free quality score and frame selection

For each candidate frame, the quality score is computed from image statistics, without using the class label:

\[
q_t = 0.50\,r_t + 0.25\,c_t + 0.25\,s_t,
\]

where \(r_t\) is valid-region fraction, \(c_t\) is robust contrast, and \(s_t\) is robust log-sharpness. The quality-selected four-frame procedure divides the sequence into temporal bins, selects the highest-quality frame in each bin, and returns selected indices in temporal order. This enforces temporal coverage while preferring clearer frames within each bin. The precise image preprocessing and sampling parameters are recorded in the resolved run configuration.

### B1 static anchor

B1 uses one frame's frozen feature vector and a logistic-regression classifier. Hyperparameter selection is performed within training data (the current candidate grid is \(C \in \{0.01, 0.1, 1, 10\}\)); scaling and fitting are restricted to the relevant training fold. B1 is the static reference for assessing whether additional temporal observations and aggregation are useful.

### Constrained MIL-4 and final calibration

The MIL model receives four selected frame vectors. A projection block maps each frame to a 96-dimensional hidden representation, followed by gated attention:

\[
 h_t = \mathrm{GELU}(\mathrm{LayerNorm}(W x_t+b)),\qquad
 a_t \propto \exp\{w^\top[\tanh(Vh_t)\odot\sigma(Uh_t)]\},\qquad
 z = \sum_t a_t h_t.
\]

A small classifier maps the pooled representation \(z\) to a video probability. The constrained final calibration combines the B1 and MIL probabilities,

\[
 p_{\mathrm{fused}}=(1-\lambda)p_{B1}+\lambda p_{MIL},
\]

with Platt calibration fitted from inner out-of-fold predictions. The current candidate grid for \(\lambda\) is \(\{0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50\}\). Candidate selection uses inner-fold results subject to accuracy and Brier constraints relative to B1; the outer fold is reserved for evaluation. See the resolved configuration and calibration script for the exact tie-breaking and threshold-selection rules. A selected value at the edge of a search grid should be treated as a prompt for further validation, not evidence of a globally optimal fusion weight.

### Ablation controls

- **Quality-4 mean:** select the same quality-guided, temporally distributed four frames, average their features, and apply the same calibration protocol.
- **Uniform-4 mean:** select four uniformly spaced frames, average their features, and apply the same calibration protocol.
- **B1 + calibration (λ = 0):** control for the calibration procedure without adding MIL information.

These controls help separate the effects of adding frames, selecting frames by quality, learning attention weights, and calibrating/fusing predictions. Their conclusions remain limited to the current dataset and evaluation protocol.

## Data and evaluation protocol

The current experiment contains **128 animal-level videos** with one video per animal and binary labels (97 positive, 31 negative). The files and labels are supplied outside this repository. The research dataset's local directory naming may refer to a disease cohort; readers should consult the approved data documentation for the exact label definition. These results do not establish clinical diagnostic performance.

Evaluation uses a five-fold stratified group cross-validation split (`sgkf_v1`, seed 42) and four inner folds for model selection/calibration. The current reported table represents one outer-CV repeat. All compared methods should use the same split manifest and training-only rules. Reported metrics are ROC-AUC, average precision, accuracy, balanced accuracy, and Brier score; threshold-dependent metrics depend on the threshold-selection procedure. Because the cohort is imbalanced, accuracy alone is not sufficient.

The repository's run artifacts include split and feature manifests and resolved configurations. They are useful for auditability, but may contain machine-specific paths or dataset identifiers and should be reviewed before publishing.

## Repository layout

This is a research repository whose exact contents may evolve. Key entry points currently include:

| Path | Purpose |
|---|---|
| `extract_features_v1.py` | Extract frozen frame features and write a feature manifest/cache. |
| `train_b1.py` | Run the B1 single-frame baseline. |
| `train_b1_V2.py` | Run the current frame aggregation experiments, including `mil_4`. |
| `train_b1_mil4_final_calibration.py` | Fit/evaluate constrained B1–MIL probability fusion and final calibration. |
| `ablate_mil4_mean_and_quality_single.py` | Run quality-4 or uniform-4 feature-mean ablations. |
| `runs/B1_B5/` | Current five-method comparison artifacts and fold predictions. Review before public release. |
| `config_v1_local.yaml` | Local configuration template/configuration; paths and data settings are machine-specific. |

Other experiment folders contain exploratory or historical results. Their presence does not mean every experiment is part of the primary validated comparison.

## Experimental program: B1–B9 and P1–P3

The repository contains two related but distinct experimental tracks. **B1–B9** are comparison baselines and architectural probes; **P1–P3** are the graph-temporal model progression described in the research task brief. The five-method few-frame table above is a later, separate RetinaSAGE ablation on `runs/B1_B5`; do not combine its values with the historical B/P summary below. The latter reports fold mean ± standard deviation in the 2026-10-13 evaluation note, and those standard deviations are not confidence intervals.

| Experiment | Main idea | Reported result / status in the 2026-10-13 evaluation |
|---|---|---|
| **B1** | Frozen DINOv3 feature from one fixed frame + logistic regression; static reference baseline. | AUC 0.9733 ± 0.0596; accuracy 0.9846 ± 0.0211. Strongest reported classification baseline in that comparison. |
| **B2** | Average sampled frames in pixel space, encode the resulting image with DINOv3, then logistic regression. Tests whether temporal pixel averaging helps. | AUC 0.9625 ± 0.0650; accuracy 0.9535 ± 0.0501. Below B1 in the reported summary. |
| **B3** | Encode frames independently, average frozen DINOv3 features, then logistic regression. Tests feature-space averaging without a learned temporal model. | AUC 0.9666 ± 0.0699; accuracy 0.9535 ± 0.0833. Below B1 in the reported summary. |
| **B4** | Deep Sets over the sequence of frozen global DINOv3 features; permutation-invariant learned aggregation. | AUC 0.9549 ± 0.0822; accuracy 0.9455 ± 0.0436. |
| **B5** | GRU probe over the sequence of frozen global DINOv3 features; tests order-sensitive recurrent aggregation. | AUC 0.9461 ± 0.0828; accuracy 0.9142 ± 0.0687. |
| **B6** | Local-region temporal Transformer with the spatial GAT module disabled; matched ablation for assessing the graph module. | AUC 0.9248 ± 0.0728; accuracy 0.9062 ± 0.0219. The report says P1 did not show an AUC gain over this no-GAT comparison. |
| **B7** | No B7 experiment is identified in the available evaluation note or current named training entry points. | Not reported here; the label is intentionally left unmapped pending its original specification/artifacts. |
| **B8** | RETFound-MAE image representation with mean pooling and a logistic-regression probe. | AUC 0.9247 ± 0.1036; accuracy 0.9058 ± 0.0598. |
| **B9** | Frozen V-JEPA 2.1 video representation with a lightweight probe. | AUC 0.9222 ± 0.0852; accuracy 0.8594 ± 0.0439. Exact token pooling should be checked against the run configuration before reproducing. |
| **P1** | Local region graph + GATv2 spatial aggregation + causal temporal Transformer; classification objective only. | AUC 0.9218 ± 0.0669; accuracy 0.9065 ± 0.0579. The report found no classification gain over B6 in that run. |
| **P2** | P1 plus a future local-feature prediction loss at configured temporal horizons. | AUC 0.9269 ± 0.0648; accuracy 0.8757 ± 0.0825. AUC was higher than P1 in the summary, while accuracy was lower; predictive quality needs its own horizon-wise evaluation. |
| **P3** | P2 plus fixed-reference, multiscale Frangi structure weighting and a reliable-region mask. | AUC 0.9477 ± 0.0689; accuracy 0.8988 ± 0.0438. The report describes this as an improvement within the region-temporal model family, while still below B1's reported AUC. |

The P-series results do not establish that temporal order, GATv2, or future prediction improves over a strong static baseline. P3 changes both structure weighting and region reliability handling; the report therefore treats their combined effect as a candidate signal rather than attributing the full change to Frangi alone. The separate P3 ablation runners support removing Frangi weighting or the reliable-region mask. These are research comparisons, not clinical validation.

### Additional code map

| Code | Role |
|---|---|
| `dataset_v1.py` | Configuration, manifest, and dataset-loading utilities shared by training pipelines. |
| `model_p123.py` | Main P1–P3 graph-temporal network, including spatial graph attention, causal temporal processing, and prediction heads. |
| `train_b12345.py` | Shared nested-CV implementation for B1–B5 (single frame, pixel mean, feature mean, Deep Sets, and GRU); `train_b1.py` and `train_b1_V2.py` provide task-specific entry points for baseline/few-frame work. |
| `train_b6.py`, `train_b89.py` | No-GAT B6 comparison and B8/B9 representation probes. |
| `train_p123.py` | Shared nested-CV runner for P1, P2, and P3. |
| `train_p1_xr.py`, `train_p2_xr.py`, `train_p3_xr.py` | Extended matched evaluation/ablation runners, including P3 structure and mask ablations. These are related implementations; use the exact runner/config recorded for a result rather than assuming all scripts are interchangeable. |
| `extract_features_v1.py`, `extract_features_v2.py`, `extract_retfound_features.py`, `extract_V_JEPA_features.py` | Feature extraction and caching for DINOv3 and alternative pretrained encoders. Checkpoint access and cache compatibility are configuration-dependent. |
| `future_evaluation.py` | Future-horizon matching and evaluation utilities used by predictive experiments. |
| `infer_v1.py` | Inference entry point for the applicable trained model/configuration. |
| `ablate_mil4_mean_and_quality_single.py`, `train_b1_mil4_final_calibration.py` | RetinaSAGE few-frame mean ablations and B1–MIL-4 calibration/fusion. |
| `train_b1_*`, `analyze_*.py`, `compare_*.py` | Additional targeted research branches (residual/PCA/spatial/cost-sensitive probes, calibration analysis, and comparisons). They are exploratory unless explicitly included in a reported table. |
| `config_*.yaml`, `requirements.txt` | Experiment settings and Python dependencies. Local configuration files may contain machine-specific data/checkpoint paths. |
| `vjepa2/` | Related V-JEPA 2 code/resources; it is not required for the primary frozen-DINOv3 RetinaSAGE run. |

## Reproducibility

### Environment

The project currently targets Python with PyTorch and scikit-learn. Install dependencies in an isolated environment and adapt versions to the available CUDA driver and platform:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The feature pipeline also requires access to the DINOv3 checkpoint and the data. Neither is redistributed by this README. Checkpoint access and use are subject to their respective terms.

### Configure local data

Create a local configuration based on the project's YAML configuration and set paths for the video data, label workbook, feature cache, and DINOv3 checkpoint. Do not commit private data paths or credentials. A typical configuration includes:

```yaml
# Illustrative keys only: follow the schema in the repository configuration.
data:
  # video roots and label source
features:
  # cache directory and DINOv3 checkpoint
experiment:
  # frame sampling rate, maximum frames, image size, seed, and output directory
```

Use the same configuration, split manifest, feature manifest, and frozen cache for all methods in a comparison. Preserve the resolved configuration with each run.

### Commands

The following commands illustrate the current script interfaces; run `--help` first if the interface has changed. Replace the placeholder paths with local, accessible files:

```bash
# Extract/cache frozen frame features (omit this step when reusing a verified cache).
python extract_features_v1.py \
  --config /path/to/local_config.yaml

# B1 baseline.
python train_b1.py \
  --config /path/to/local_config.yaml \
  --feature-manifest /path/to/feature_extraction_manifest.csv \
  --split-manifest /path/to/split_manifest.csv \
  --device cuda

# Four-frame gated-attention MIL.
python train_b1_V2.py \
  --config /path/to/local_config.yaml \
  --feature-manifest /path/to/feature_extraction_manifest.csv \
  --split-manifest /path/to/split_manifest.csv \
  --method mil_4 \
  --device cuda

# Quality-guided or uniformly selected four-frame feature-mean ablation.
python ablate_mil4_mean_and_quality_single.py \
  --config /path/to/local_config.yaml \
  --feature-manifest /path/to/feature_extraction_manifest.csv \
  --split-manifest /path/to/split_manifest.csv \
  --budget 4 \
  --selection quality \
  --output /path/to/output_quality4

# Repeat the ablation with --selection uniform and a distinct output directory.
```

Final calibration consumes the completed B1 and MIL-4 run directories; consult `python train_b1_mil4_final_calibration.py --help` for the exact path arguments. Preserve outer-fold predictions and fold metrics. Do not select hyperparameters, calibration parameters, or thresholds using outer-test labels.

### Reproducibility checklist

For a defensible comparison, record the source revision, package versions, hardware, random seeds, resolved configuration, split-manifest hash, feature-manifest hash, checkpoint identity, and command line. Confirm that all methods use identical outer folds and that cached features correspond to those exact videos and preprocessing settings. Report both fold-level results and their aggregation rule; do not mix fold-wise means with pooled out-of-fold metrics.

## Limitations and next steps

1. **Small, imbalanced, single-cohort data.** The reported 128 videos and 31 negative examples limit precision, especially for subgroup and threshold-dependent conclusions.
2. **No independent validation.** Five-fold cross-validation is internal validation, not external replication. A second cohort/site and, ideally, a separately locked test set are needed.
3. **One CV repeat.** Fold assignment variability has not been characterized through repeated nested cross-validation or bootstrap analysis.
4. **Quality-selection effect is unresolved.** In the current table, uniform-4 mean has higher AUC and average precision than quality-4 mean. Quality selection therefore needs further testing across datasets/tasks before claiming a general advantage.
5. **Calibration and threshold uncertainty.** The fusion and threshold rules are trained within inner folds, but their stability across repeats and cohorts remains to be established. Grid-boundary selections need targeted follow-up.
6. **Compute and latency reporting.** Frozen feature caching reduces repeated backbone work, but full end-to-end latency, memory, and storage costs have not been comprehensively benchmarked. Report encoding and per-video aggregation costs separately for deployment-oriented claims.
7. **Scope of the label.** The binary study label and cohort definition should be reported under the governing data protocol. The model is a research classifier and must not be presented as a clinical decision tool.

Planned extensions include repeated nested validation, external-cohort and cross-task evaluation, uncertainty intervals, robustness to frame sampling and image quality, and measured inference cost under a specified deployment setup. Historical negative results and exploratory branches should remain available for transparent research accounting.

## Data, privacy, and artifact policy

The dataset, labels, and pretrained checkpoints are not included. Before publishing this repository, inspect files and Git history for raw videos, images, label workbooks, identifiers, absolute local paths, credentials, and restricted model weights. Publish only artifacts that are cleared for redistribution. Prefer sanitized configuration templates and aggregate metrics; share fold-level predictions only when the data governance approval explicitly permits it. Add local caches, private manifests, and large run outputs to `.gitignore` as appropriate.

## Citation

If you use this code or the experimental protocol, cite the repository after its archival release and cite the underlying methods as appropriate. Relevant methodological references include:

- Siméoni et al. **DINOv3**. [arXiv:2508.10104](https://arxiv.org/abs/2508.10104).
- Ilse, Tomczak, and Welling. **Attention-based Deep Multiple Instance Learning**. [PMLR 80 (2018)](https://proceedings.mlr.press/v80/ilse18a.html).
- Zaheer et al. **Deep Sets**. [NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html).
- Brody, Alon, and Yahav. **How Attentive are Graph Attention Networks?** (GATv2). [arXiv:2105.14491](https://arxiv.org/abs/2105.14491). This is relevant to exploratory graph branches; GATv2 is not used by the primary MIL-4 result above.

## License

Add the license governing this code before redistribution. Data, model weights, and third-party dependencies may have separate terms; a code license does not grant rights to redistribute them.

Copyright (c) 2025 Taotao9029. All rights reserved.

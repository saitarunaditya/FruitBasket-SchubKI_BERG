# Report - Mind Reading from EEG (FruitBasket, SchubKI Hackathon 2026)

## 1. Problem

Given a 3-second EEG recording (17 channels, 500 Hz) made while a subject watched a
short video, rank 102 candidate videos from most to least likely to have produced that
brain activity. Scored by top-1 and top-5 accuracy on held-out subjects sub-05/06.

## 2. Approach

We frame it as an **encoding** problem (stimulus -> predicted brain response), then
identify a recording by matching its measured EEG against the predicted EEG of every
candidate video. This mirrors the organizer baseline but swaps the stimulus
representation.

Common pipeline:
1. Decimate EEG 500 -> 100 Hz (1500 -> 300 samples), anti-alias filtered.
2. Sample 8 frames per video.
3. Turn each video into a feature vector (the part we vary across experiments).
4. `StandardScaler -> PCA(0.95) -> StandardScaler -> RidgeCV` maps features -> real EEG.
5. Predict EEG for each candidate; rank by z-scored correlation against measured EEG
   on the window [5:60].

## 3. Experiments and results (subject 1 unless stated)

| # | Feature source | Training | top-1 | top-5 | median rank |
|---|---|---|---|---|---|
| 02 | AlexNet `features.3` | sub-01 | 0.0392 | 0.1176 | 25.5 |
| 03 | BERG in silico EEG, used directly (Option A) | none (pretrained) | 0.0196 | 0.0784 | - |
| 04 | BERG in silico EEG as features (Option B) | sub-01 | **0.0490** | **0.2157** | **20.0** |
| 07 | AlexNet `features.3`, pooled | all 6 subjects | 0.0392 | 0.1176 | - |
| 06 | BERG intermediate ViT CLS feature | all 4 dev subjects | run to measure | run to measure | - |

Chance: top-1 0.0098, top-5 0.0490, median rank ~51.5.

Notebook 07 per-subject (pooled AlexNet): sub-01 0.0392/0.1176, sub-02 0.0588/0.1569,
sub-03 0.0588/0.1765, sub-04 0.0294/0.0980; mean 0.0466 / 0.1373.

## 4. What won and why

**Option B (BERG features + per-subject Ridge, notebooks 04/05) is the chosen
submission.** It gave the best measured result: top-5 0.2157 is ~1.8x the AlexNet
baseline and ~4.4x chance, with the best median rank (20.0).

Key insight: BERG's raw in silico EEG matched *directly* (Option A) is weak (0.0196 /
0.0784) because the model was trained on still images in a 63x140 layout, unlike our
movie EEG (17x300). But used as a *stimulus descriptor* feeding a subject-trained Ridge
encoder, the same representation outperforms AlexNet. The learned mapping absorbs both
the image-vs-movie gap and the layout mismatch.

## 5. Failure cases and lessons learned

- **Option A (direct BERG) underperformed** - trusting a pretrained encoder's absolute
  output across domains (images -> movies) does not transfer; a thin learned adapter does.
- **Pooling subjects helped only modestly for AlexNet** (nb 07): mean top-5 0.137 vs
  single-subject 0.118, and it did not change sub-01 at all. More cross-subject data is
  not automatically better when subjects differ.
- **Data coverage gap:** the AlexNet feature file only covers the 968 videos sub-01 saw;
  other subjects each had ~30 videos without features, which nb 07 drops (~3% loss).
- **Windows/UTF-8:** BERG crashed on import (cp1252 vs UTF-8 YAML). Fixed with a UTF-8
  `open()` patch / `PYTHONUTF8=1`.
- **Filename formats:** training videos are zero-padded (`0012.mp4`), test are `1001.mp4`;
  the video lookup must handle both.
- **Submission ordering:** for sub-05/06 the ranked rows must stay in the original
  unsorted `tst` order (no label sorting, which only applies to validation).

## 6. Compliance notes

- No test data was used for training. sub-01..04 test labels were used only for
  validation/model selection. sub-05/06 were ranked using only their own training data
  (encoder) and their own test recordings (unsupervised correlation) - no cross-subject
  test templates.
- Unsupervised z-scoring of features/EEG only.
- Public pretrained models: BERG `eeg-things_eeg_2-vit_b_32` (THINGS-EEG2, ViT-B/32);
  AlexNet `features.3` via net2brain. Both released before the hackathon.

## 7. Reproducing

See the root `README.md`. The submission is produced by `05_make_submission.ipynb`,
which also runs a sub-01 format+score self-check before writing the sub-05/06 files.

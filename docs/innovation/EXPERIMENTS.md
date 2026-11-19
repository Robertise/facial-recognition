# Individual innovation: adversarial eyeglasses against the attendance system, and a certified defense

This file is the experiment record of the individual innovation. It states, for every stage, the question, the decision rule fixed **before** the numbers existed, the result, and what we concluded. Every number names the run folder in `results/innovation/` that produced it; each run's exact command line is stored in its `meta.json`.

Code: `innovation/`. How to run each stage: `innovation/README.md`.

## Contents
1. Threat model and setup
2. Attack: adversarial eyeglasses (digital and physical)
3. Mechanism: where the fake identity lives
4. Defense 1: purifier (erase the glasses band, inpaint)
5. Adaptive attack against the purifier
6. Defense 2: certified multi-mask consensus (v1, v2, v3)
7. Held-out test of the certified defense
8. Summary table
9. Limitations
10. Directions considered and rejected

---

## 1. Threat model and setup

**System under attack.** The team's attendance pipeline: MediaPipe face detection, the team's liveness model, low-light enhancer, and the face recognizer `method3` (FaceCNN + ArcFace, 512-d embedding, cosine similarity, accept threshold 0.5, 1:N match against the enrolled gallery). Nothing in the team pipeline was changed for the attack.

**Attacker.** The author (`owner`) wears eyeglasses whose frame carries an optimized pattern, and tries to be recognized as a specific enrolled victim (impersonation). White-box: the attacker knows `method3`. The pattern is optimized with expectation over transformation (EOT: pose, scale, brightness, blur, small shifts) on webcam frames of the attacker.

**Gallery.** 53 identities with 5 photos each, built from LFW by `innovation/tools/build_gallery.py`: the victim (`LFW_victim`, chosen at the median of owner-to-candidate similarity so it is neither easy nor hard), 2 extra victims, and 50 distractors (a simulated class). LFW identities whose names appear among the 4 000 VGGFace2 training identities of `method3` were excluded, plus ArcFace member-inference filtering (`data/innovation/gallery/_build_report.json`). The owner and a second real person (`p01`) are enrolled too (`p01` in `gallery_people`).

**Sessions (webcam, 75 frames each).** Owner: `s1_day1`, `s2_day2` (attack optimization and dev), `s3_vary`–`s5_vary` (varied distance/pose; dev for the defense), `s6_vary` (held-out evaluation for every attack and defense). `p01`: `s1` (no glasses; not recognized even clean, so unused), `s2` (held-out genuine set for the defense).

**Metrics.** Attack success = fraction of frames that pass liveness and are accepted as the victim. GAR = fraction of genuine frames accepted as the right person (liveness gated on test). 95% intervals are Wilson intervals. Each configuration is scored on its held-out set exactly once.

**Protocol.** Every stage had a written decision rule (gates, priority order, thresholds) logged before any number of that stage existed. Rejected options and their reasons were logged so they are not re-run. When a measurement bug was found, the affected runs were voided and rerun; voided runs are not used anywhere below.

---

## 2. Attack: adversarial eyeglasses

### 2.1 Pre-registered gate
A configuration passes if its success on the held-out session is at least 80%, with both controls (random pattern, plain frame) at 0%. Configurations are tried in a fixed priority order (most wearable first); the first that passes on dev is chosen.

### 2.2 First attempts (elliptical frame)
- Round 1 (`20261104_*`): no configuration reached the gate.
- Round 2 ladder R2a/R2b/R2c (`20261106_130834_ladder`): R2b chosen by rule (dev 93%). Held-out `s6_vary`: **R2b 62/75 = 82.7%**, controls 0 (`20261106_192143`). Sessions `s3`–`s5` were rejected by a pre-registered baseline check (liveness too low on clean frames), not by attack results.
- Finding: R2b's frame closes almost the whole eye area (lens shrinks to a 2 px slit). It is not wearable glasses, so the work moved to the owner's real frame.

### 2.3 Real frame (H-ladder)
- Mask from a photo of the owner's real square frame (`innovation/tools/frame_mask_from_photo.py`, `data/innovation/frame/`), scaled by IPD 63 mm. The pattern covers rim and lens and leaves a clear disc around each pupil of diameter 16/13/11/9 mm (H16/H13/H11/H09).
- Dev (`20261107_210423_ladder`, rerun after an earlier run was voided for a train/held-out split leak): no configuration reached 80%; by rule the highest (H09, 60%) was chosen.
- Held-out `s6`, each scored once (`20261108_2151*`–`2156*`): H16 20.0%, H13 24.0%, H11 69.3%, **H09 65/75 = 86.7%** (gate passed, controls 0).
- Seeds (rule: seed 0 is the result, seeds 1–2 are description only): seed 1 69.3%, seed 2 80.0%. **H09 succeeds on 69–87% of held-out frames depending on the seed.**
- Transfer to the team's other recognizers is weak: method1 0%, method2 6.7%, method4 0%, ViT (method5) 0%. The attack is specific to the white-box model.

### 2.4 Physical attempt (one time-boxed try, pre-registered)
- Colour chart printed and photographed through the webcam (`data/innovation/chart/`): printable colours only span about 0.06–0.69 of each channel.
- H09 re-optimized under the measured colour map (`20261109_094721`): **10.7%** on `s6` digitally.
- Printed glasses, recorded live (`20261109_212933_advglasses_eval_physical`): **4/75 = 5.3%**, controls 0. Gate failed; printing was closed by the rule.
- Chain for the report: **87% (digital, perfect colour) → 11% (printable colours) → 5% (physical)**. The bottleneck is the printable colour range, not liveness or geometry. All later defense work uses digital attacks.

---

## 3. Mechanism: where the fake identity lives
Rule (`20261106_210957_signatures`): a candidate signature is kept if AUC (attack vs genuine) ≥ 0.90, secondary if 0.75–0.90, dropped if < 0.75.

- **S1** (residual true identity: top-2 gap/ratio): AUC 0.62, dropped.
- **S3** (temporal instability of top-1): AUC 0.68, dropped. Successful attacks are stable frame to frame.
- **S2** (mask 8 horizontal bands one at a time, measure the score drop): AUC 0.997, kept. Masking the eye/frame band removes about 75% of the fake match, while genuine faces drop about 22% spread over all bands.
- S2 holds on the real frame H09 and against a second real person (`20261110_212841_signatures`): AUC 0.9999 vs `p01`, 0/469 false positives at 95% TPR. It does have 14% false positives on LFW photos.
- The H-ladder adds a second piece of evidence: success jumps from 24% to 69% when the clear disc shrinks from 13 mm to 11 mm.

**Conclusion:** the fake identity is concentrated around the eyes, exactly under the glasses. This motivates defenses that remove that region rather than detect the pattern.

---

## 4. Defense 1: purifier
**Design.** Always erase a fixed "band" (the union of every attacked frame shape, dilated; 17% of the 112 px crop) and repaint it with a small U-Net (3.7 M parameters) trained with an identity loss through frozen `method3` on LFW only (11 388 train / 1 206 val, identities disjoint from the gallery). Gallery images go through the same purifier. If eye landmarks are missing, the frame is rejected. A pattern inside the band cannot affect the output, by construction (unit-tested).

**Rule** (before code existed): (1) attack success ≤ 10% for H09 and R2b on `s6`; (2) GAR ≥ 0.90 × GAR(no defense) on `owner/s6` and `p01/s2`; "better than gray fill" if GAR is ≥ 5 points higher on both sets or gray fails (2). Checkpoint chosen on LFW val only.

**Result** (main `20261112_103208_purifier_eval`, purifier `20261112_004047`; seeds 1–2 `20261112_160724/160820/173637`):

| | no defense | gray fill | purifier seed 0 / 1 / 2 |
|---|---|---|---|
| H09 ×3 seeds, R2b | 69–87% | 0% | **0% / 0% / 0%** |
| GAR owner/s6 | 97.3% | 86.7% | 90.7 / 94.7 / 89.3% |
| GAR p01/s2 | 97.3% | 94.7% | 96.0 / 96.0 / 96.0% |

Passes on all 3 seeds. Honest reading: most of the blocking comes from erasing the band at all (gray fill also blocks 0%). The learned part adds only a few GAR points on webcam data, although on LFW val it raises identity cosine from 0.39 (gray) to 0.74. A consistency detector (identity before vs after purification) separated attacks perfectly here (AUC 1.0, fixed threshold 0.4253), but only against a non-adaptive attacker.

---

## 5. Adaptive attack against the purifier
**Question.** What does an attacker who knows the whole pipeline have to wear to get through?

**Setup.** The pattern goes on a ring of width r outside the band, r ∈ {8, 16, 32, 48} px template (2.8 / 5.6 / 11 / 17 mm), optimized jointly through the no-defense and purifier branches (`innovation/defense/adaptive.py`). Score `success_full` = accepted as victim through the purifier, with liveness, and not flagged by the consistency detector.

**Rule.** (a) The defense holds within the "glasses" threat model if `success_full` ≤ 20% at both r = 8 and r = 16. (b) The attacker's cost is the smallest r with `success_full` ≥ 50%. A validity check on the measurement must pass before reading a ring. A first set of runs (12/11) had a gallery measurement bug and was voided before reading.

**Result** (optimize `20261113_074527/074700/074829/075000`, eval `20261113_173745/173823/173856/175051`):

| r (mm) | band + ring covers | success through purifier | flagged |
|---|---|---|---|
| 8 (2.8) | ~20% of crop | 0/75 (validity check failed, not read) | 0 |
| 16 (5.6) | ~23% | 0/75 | 0 |
| 32 (11) | ~29% | 1/75 | 0 |
| 48 (17) | ~35% | **42/75 = 56%** [45–67%] | **0** |

- (a) cannot be confirmed by the rule, because r = 8 failed its validity check. r = 16 is 0/75.
- (b) The attacker's cost is a 17 mm ring.
- The consistency detector is useless against this attacker: it flagged no frame at all.
- At r = 48 the purifier even makes the attack easier (56% vs 44% with no defense).

**Conclusion:** any empirical defense gets broken by the next adaptive attacker. That led to a defense whose guarantee does not depend on the attack algorithm.

---

## 6. Defense 2: certified multi-mask consensus

### 6.1 Idea
Define a set S of masks, anchored on the two eye centres:
- **Glasses family G_r:** the band grown by r + δ, with r ∈ {0, 16, 32, 48} and δ = 8 px.
- **L:** the lower face (face masks, scarves).
- **F:** the forehead (hats).

For each mask m in S, one branch: crop, grey out m, enhance, fill (gray or a learned inpainter), recognize, match 1:N against a gallery processed the same way. **Accept identity X only if every branch says X with similarity ≥ 0.5.**

If an accessory lies entirely inside one region m*, branch m* only sees the clean face. So the attacker can only be accepted as V if its own clean face would be accepted as V. This holds for every attack algorithm, assuming the eye landmarks move by at most δ and the accessory sits in one region. It is a one-sided certificate ("cannot impersonate"), which is what attendance needs: a rejected genuine user simply retries.

Related work: first round of PatchCleanser (USENIX Security 2022), Minority Reports (2020). What is different here: landmark-anchored masks per accessory family, an asymmetric certificate for thresholded 1:N matching, and a learned fill that keeps the certificate.

### 6.2 Selection rule (dev) and test rule
- **Dev rule:** a set S passes if its GAR ratio to no defense is ≥ 0.90 on owner dev (`s3`–`s5`) and passes the LFW condition. In v1/v2 the LFW condition was ≥ 0.90 on LFW val. In v3 it was no worse than v1 on LFW val B, chosen before v3 numbers existed.
- **Priority order:** all three families {G_r, L, F} with r = 48 → 0, then two families, then G_r alone.
- **Test rule** (run once): GAR(S) ≥ 0.90 × GAR(none) on `owner/s6` and `p01/s2`. Every attack inside the certified region must succeed on 0% of frames; any success means an implementation bug.
- Dev used for tuning is optimistic by construction. The conclusion rests only on the test run.

### 6.3 v1: one inpainter for all masks (failed dev)
- **Inpainter** `20261115_144634`: LFW val cosine, inpaint vs gray: G0 0.71/0.37, G48 0.60/0.25, L 0.70/0.41, F 0.83/0.63.
- **Dev:** no set passed (`20261115_230858`, `20261115_222002`). {G0} reached 0.82 on owner and 0.60 on LFW.
- **Diagnosis:** the main error was matching the wrong person when probe and gallery are masked in the same place, not the threshold.

### 6.4 v2: student recognizer + clean gallery (failed, worse than v1)
- **Method:** a student recognizer initialized from `method3` and distilled to map masked faces to the teacher's clean embedding (cosine + InfoNCE), compared against an unmasked gallery, with optional FAR-calibrated thresholds.
- **Result:** no configuration passed, and none beat v1. Comparing masked to clean is worse than masking both sides (owner {G0}: 0.84 symmetric vs 0.56 clean).
- **What it showed:** the student does learn to read masked faces (0.56 → 0.79 on the clean gallery), but it was used in the wrong comparison mode. LFW has mostly one image per identity, which is too little data.

### 6.5 v3: symmetric student + more data (passed dev)
- **C7:** the v2 student used symmetrically, with the gallery masked and encoded by the same student. No new training.
- **C8:** a new student (`20261118_135729`, initialized from v2), trained on LFW train plus the VGGFace2 test split. That is 472 identities and 28 281 face crops, sharing no identity with `method3`'s training set, LFW val or the gallery (`innovation/defense/prepare_faces.py`). Loss: distillation plus a supervised symmetric contrastive term (two different photos of the same person, same mask, must match). Batches sample identities in proportion to their image count.
- **C9:** C8 plus calibrated thresholds.

Ratio to no defense, fill inpaint, owner dev / LFW val B (runs listed in §11):

| | {G0} | {G16} | {G48} | {G0,F} | {G16,F} | {G0,L,F} | {G48,L,F} |
|---|---|---|---|---|---|---|---|
| v1 (C0) | 0.84 / 0.70 | 0.75 / 0.66 | 0.39 / 0.61 | 0.83 / 0.50 | – / 0.48 | 0.70 / 0.36 | 0.34 / 0.36 |
| C7 | 0.97 / 0.77 | 0.90 / 0.75 | 0.55 / 0.61 | 0.97 / 0.54 | 0.90 / 0.54 | 0.87 / 0.42 | 0.53 / 0.36 |
| **C8** | 0.93 / 1.22 | 0.92 / 1.20 | 0.72 / 1.04 | 0.93 / 1.01 | **0.92 / 0.96** | 0.90 / 0.85 | 0.70 / 0.75 |
| C9 | 0.91 / 0.89 | 0.87 / 0.77 | 0.42 / 0.63 | 0.91 / 0.69 | 0.87 / 0.61 | 0.84 / 0.52 | 0.38 / 0.39 |

**Selected by rule: C8, S = {G16, F}** (certified radius r* = 16 px, about 5.6 mm beyond the band, plus hats).
- It also passes the stricter v1/v2 LFW condition. C8 {G0,L,F} missed by 0.0036 on owner.
- **What the gain comes from:** most of the owner gain comes from using the student symmetrically (C7, no new training: {G0} 0.84 → 0.97). The extra data and contrastive loss mainly help wide glasses (G32 0.73 → 0.88, G48 0.55 → 0.72) and LFW.
- **LFW ratios above 1** happen because the masked branches use the student while the no-defense branch uses `method3`. They show a better recognizer, not a free certificate.
- Calibrated thresholds (C9) made things worse.

---

## 7. Held-out test of the certified defense (run once)
Run `20261119_064729_certified_eval`, laptop CPU, liveness on.

| | owner/s6 | p01/s2 |
|---|---|---|
| GAR, no defense | 97.3% | 97.3% |
| GAR, S = {G16, F} | 96.0% (72/75) | 97.3% (73/75) |
| Ratio | **0.986** | **1.00** |
| Certified rate (no branch matches anyone else) | 98.7% | 100% |

| Attack (pasted on owner/s6) | no defense | purifier | **certified {G16, F}** |
|---|---|---|---|
| H09 seed 0 / 1 / 2 | 86.7 / 69.3 / 80.0% | 0% | **0 / 0 / 0%** |
| R2b | 82.7% | 0% | **0%** |
| Adaptive ring r = 8 / 16 | 0 / 0% | 0 / 0% | **0 / 0%** |
| Adaptive ring r = 32 / 48 (outside the certificate) | 0 / 44% | 1.3 / 56% | **0 / 0%** |

- **Test rule passed** on both genuine sets, with no success inside the certified region.
- ADA r = 48 also drops to 0%. That is an observed result, not a guarantee: the pattern was optimized against the purifier, not against S.
- **Learned fill vs gray** on the chosen set: inpaint 96.0 / 97.3% vs gray 97.3 / 94.7%. The pre-registered "better than gray" condition (≥ 5 points on both sets) is **not met** for {G16, F}. The learned fill matters for wide masks: on the same test, {G48, L, F} keeps 0.986 × with inpaint but drops to 0.56 × with gray.
- **Landmark assumption.** The attacks themselves move the detected eye centres:

  | Attack | Mean / p95 shift (px template, IPD 180) | Frames above δ = 8 |
  |---|---|---|
  | H09 s0 | 12.0 / 16.4 | 70/75 |
  | R2b | 24.0 / 30.7 | 75/75 |
  | ADA r16 | 5.4 / 9.2 | 11/75 |

  A pattern lying r_p px outside the band is still covered by G16 if the shift is at most 16 + 8 − r_p. Frames inside the formal guarantee: H09 75/75 (each seed), ADA r8 75/75, ADA r16 64/75, R2b 39/75. The 0% on the remaining frames is empirical.
- **The test sessions are easier than dev** (owner/s6 is recognized on 100% of frames with no defense). On test, every glasses set keeps ≥ 0.97 × with inpaint. The dev table in §6.5 is the honest picture of what certification costs; the test confirms only the pre-registered choice. With 75 frames per set, the lower 95% bound of GAR is about 0.89.

---

## 8. Summary table (held-out `owner/s6` unless noted)

| | GAR owner | GAR p01 | H09 (3 seeds) | R2b | Adaptive 17 mm ring | Guarantee |
|---|---|---|---|---|---|---|
| No defense | 97.3% | 97.3% | 69–87% | 82.7% | 44% | none |
| Gray fill of band | 86.7% | 94.7% | 0% | 0% | (not targeted) | none |
| Purifier (band + U-Net) | 90.7% | 96.0% | 0% | 0% | **56%** | none |
| **Certified {G16, F}** | **96.0%** | **97.3%** | **0%** | **0%** | **0%** (empirical) | cannot impersonate with any accessory inside G16 or F |

---

## 9. Limitations
- **Digital attacks only.** The physical print reached 5%, so the defense was never tested against worn printed glasses.
- **One attacker** (the author) and one target victim. The second real person is used only as a genuine user.
- 75 frames per session, so confidence intervals are about ±5–8 points.
- **The certificate rests on assumptions:** eye-landmark shift within the margin, which is violated on part of the attack frames (§7); an accessory inside one region (an accessory spanning two regions is not covered); and correct liveness.
- **Dev was used for tuning** in v2/v3, so dev numbers are optimistic. The test sessions turned out easier than dev.
- **LFW val is press photos,** a different domain from webcam frames. The LFW ratio in v3 compares the student against `method3`.
- The certified set chosen excludes wide glasses (G32/G48) and the face-mask family L. The 17 mm adaptive ring is stopped only empirically.

---

## 10. Directions considered and rejected
- **Liveness-style challenges and other cues:** screen-flash colour challenge, motion challenge, rPPG, passive blink, phone/YOLO detection, depth/CDCN, sonar, multi-face. Rejected as not novel enough, too frontend-dependent, or out of scope.
- **Voice + digit-challenge identity binding (with replay and lip-sync levels):** built in an early prototype, dropped on 5/11 as commodity work and removed from this repo. The adversarial glasses level was the only part that was new.
- **Signatures S1 and S3:** dropped by rule (§3).
- **"Blind" purifier** (learn to restore faces wearing synthetic glasses, without erasing): an adaptive attacker optimizes straight through it.
- **Training the purifier on owner/p01 frames:** leaks into evaluation.
- **Adversarial training against ring attacks:** cat-and-mouse, with no guarantee.
- **Region-voting recognizers (derandomized smoothing):** needs many recognizers trained on a large dataset; not feasible in time.
- **Full double-masking PatchCleanser:** only needed to certify correct recognition, while attendance only needs "cannot impersonate".
- **Lowering the threshold** to raise GAR: the error was wrong-person matches, so a lower threshold raises FAR and lowers the certified rate.
- **Dropping the LFW condition, or lowering the 0.90 owner threshold after seeing numbers:** this would move the goalposts.

---

## 11. Run index
- **Attack:**
  - R2b: `20261106_132526_advglasses_impersonate`, s6 eval `20261106_192143`.
  - H09: seeds `20261107_214538`, `20261112_112957`, `20261112_115550` (`_advglasses_impersonate`); s6 evals `20261108_215113`, `20261112_115503`, `20261112_122426`.
  - H-ladder dev: `20261107_210423_ladder`.
  - Physical: optimize `20261109_094721`, eval `20261109_212933_advglasses_eval_physical`.
- **Signatures:** `20261106_210957`, `20261107_124516`, `20261110_212841` (`_signatures`).
- **Purifier:** train `20261112_004047`, `20261112_043800`, `20261112_044408`; eval `20261112_103208`, `20261112_160724`, `20261112_160820`, `20261112_173637`.
- **Adaptive:** as listed in §5.
- **Certified:**
  - Inpainter `20261115_144634`.
  - v1 dev `20261115_230858`, `20261115_222002`.
  - v2 students `20261117_120646`, `20261117_123128`; v2 grid in `20261117_*_certified_*`.
  - v3 student `20261118_135729`; C7 `20261118_140609` / `140846`; C8 `20261118_141211` / `141306`; C9 `20261118_141635` / `141725` / `141819`.
  - **Test `20261119_064729_certified_eval`.**

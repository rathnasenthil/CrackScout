# CrackScout Research Protocol

# CrackScout: Selective-Segmentation Evaluation Protocol

**Version:** 1.1  
**Date:** 20 October 2026  
**Status:** Exploratory evaluation on the previously examined DeepCrack test split

## 1. Research question

Can a risk predictor based on inexpensive image-quality features improve the reliability of accepted outputs from training-free classical crack-segmentation methods, while retaining useful image coverage?

The study evaluates Otsu, adaptive Gaussian thresholding, and blackhat morphology separately.

## 2. Intended decision

For each image and segmentation method, CrackScout will either:

- **Accept:** provide the segmentation as having passed the experimental reliability screen.
- **Abstain:** withhold the segmentation and indicate that the image or result needs review.

This experiment evaluates abstention only. It does not claim to repair segmentation, automatically select another method, request recapture, or assess structural safety.

## 3. Dataset and evaluation status

Use the existing DeepCrack training split (300 image–mask pairs) and test split (237 pairs). Pair images and masks strictly within their respective split.

The test split has already been examined in previous experiments. Any new results on it will therefore be labelled **exploratory**, not confirmatory. Confirmatory validation requires an independent dataset or genuinely untouched images.

## 4. Fixed segmentation baselines

Do not change the existing baseline settings:

- **Otsu:** inverted Otsu thresholding, 3×3 opening, 5×5 closing.
- **Adaptive:** inverted adaptive Gaussian thresholding, block size 35, C=5.
- **Blackhat:** blackhat morphology, kernel size 15, threshold 25.
- **Shared preprocessing:** grayscale conversion and 5×5 Gaussian blur.

No segmentation parameters will be tuned on the test split.

## 5. Failure definition

Use image-level F1 against the ground-truth mask.

For each segmentation method, define the failure cutoff as the **25th percentile of that method’s defined image-level F1 scores on the training split**. An image is a failure if its F1 is below that method-specific cutoff.

If F1 is undefined—for example, the method predicts no crack pixels while the ground truth contains crack pixels—count the case as a failure.

Calculate and record each method’s numerical cutoff from training data before evaluating the test results. Do not change the cutoffs after viewing test performance.

This operational definition identifies relatively poor outputs within each method’s training performance; it does not establish a universal standard for acceptable crack segmentation.

## 6. Risk predictor

Use the six existing pre-segmentation quality features:

1. grayscale mean;
2. grayscale standard deviation;
3. P90–P10 intensity range;
4. Laplacian variance;
5. dark-pixel proportion;
6. local contrast.

Do not use ground-truth masks, segmentation masks, or segmentation-derived metrics as predictor inputs.

For each segmentation method, use the existing model specification: median imputation, feature scaling, and class-weighted logistic regression. Use the existing fixed random seed and grouped five-fold cross-validation procedure, keeping each image confined to one fold.

The model and preprocessing specification will not be changed after test results are inspected.

## 7. Reject-rule selection

The target is to **minimize the proportion of failures among accepted images**, subject to a minimum training-estimated coverage of **70%**.

Select the risk-score threshold using training-only out-of-fold predictions. Choose the threshold with the lowest out-of-fold failure rate among operating points that accept at least 70% of training images. If operating points tie, choose the one with greater coverage.

If no threshold meets the coverage constraint, report that the policy could not meet the pre-specified target; do not lower the target after inspecting test results.

Freeze the threshold separately for each segmentation method before computing its test performance.

## 8. Evaluation measures

For each method, report:

- **Coverage:** accepted images divided by all test images.
- **Abstention rate:** rejected images divided by all test images.
- **Selective risk:** failures among accepted images divided by accepted images.
- **Always-accept risk:** failures divided by all test images.
- **Risk reduction:** always-accept risk minus selective risk.
- **Accepted failures:** number of failures remaining among accepted images.
- **Rejected cases:** counts of rejected failures and rejected non-failures.

Also report a risk–coverage curve across thresholds, so readers can see how risk changes as coverage decreases. Include counts as well as percentages.

## 9. Comparators

Compare the learned risk policy against:

1. **Always accept:** no abstention.
2. **Random abstention:** randomly reject the same number of test images as the learned policy; repeat 1,000 times and report the distribution of selective risk.
3. **Single-feature baseline:** a risk rule using Laplacian variance alone, with its threshold selected using training data and the same 70% minimum-coverage constraint.

Failure-class precision, recall, and F1 may be reported as supplementary diagnostics, but the main evaluation is selective risk at coverage.

## 10. Reporting and interpretation

Report each segmentation method separately because their failure cutoffs and risk scores are method-specific.

Discuss the trade-off: a policy is not useful merely because it reduces risk if it does so by rejecting too many images. Describe the types of failures remaining among accepted outputs and the types of images being rejected.

Interpret findings as exploratory evidence on DeepCrack. Do not claim generalization to other datasets, cameras, surfaces, lighting conditions, or structural-safety decisions.

## 11. Follow-up validation

After the policy and thresholds are frozen, evaluate them on an independent crack-image dataset or newly collected images from different acquisition conditions. Do not recalibrate the policy on that evaluation set’s ground-truth labels before reporting performance.

## 12. Deviations

Record any change to the dataset, failure definition, feature set, model, threshold-selection rule, comparator, or metric, including its date and reason. Identify results affected by such changes as exploratory.


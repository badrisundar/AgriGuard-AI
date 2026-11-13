import pandas as pd
from pathlib import Path
from datetime import date

errors_path = Path("results/mobilenetv3_full39_val_clean_errors.csv")
errors = pd.read_csv(errors_path)

output = Path("results/AgriGuard_AI_Dataset_Model_Audit.md")
output.parent.mkdir(parents=True, exist_ok=True)

class_errors = (
    errors.groupby("actual_class")
    .size()
    .sort_values(ascending=False)
)

confusions = (
    errors.groupby(["actual_class", "predicted_class"])
    .size()
    .sort_values(ascending=False)
)

high_conf = errors[errors["confidence"] >= 0.90].sort_values(
    "confidence", ascending=False
)

avg_conf = errors["confidence"].mean() * 100

lines = [
    "# AgriGuard AI – Dataset and Model Audit",
    "",
    f"Audit date: {date.today()}",
    "",
    "## 1. Objective",
    "",
    "Evaluate the quality of the 39-class PlantVillage dataset, "
    "identify duplicate-image risks, inspect validation errors, "
    "and document limitations before further model training.",
    "",
    "## 2. Model and Dataset",
    "",
    "- Architecture: MobileNetV3 Large",
    "- Dataset: PlantVillage, 39 classes",
    "- Input image size: 224 × 224",
    "- Training images: 38,797",
    "- Original validation images: 8,300",
    "- Clean validation images: 8,297",
    "- Original test images: 8,351",
    "- Clean test images: 8,343",
    "",
    "## 3. Evaluation Results",
    "",
    "| Evaluation | Images | Accuracy |",
    "|---|---:|---:|",
    "| Original test | 8,351 | 99.58% |",
    "| Clean test | 8,343 | 99.58% |",
    "| Clean validation | 8,297 | 99.63% |",
    "",
    "Clean test metrics: Macro Precision 99.61%, "
    "Macro Recall 99.43%, Macro F1 99.52%.",
    "",
    "Clean validation metrics: Macro Precision 99.46%, "
    "Macro Recall 99.49%, Macro F1 99.47%.",
    "",
    "Note: Clean validation evaluation uses the existing checkpoint. "
    "It is diagnostic, not an independent model-selection result.",
    "",
    "## 4. Duplicate Image Audit",
    "",
    "- Exact duplicate pairs found across splits: 11.",
    "- Near-duplicate pairs (dHash distance <= 4): 228.",
    "- Train-validation exact duplicate pairs: 3.",
    "- Eight unique exact-duplicate test images were excluded "
    "to create test_clean.",
    "- Three validation images were excluded to create val_clean.",
    "- Original dataset folders were preserved.",
    "",
    "Near-duplicate matches have not all been visually verified. "
    "Therefore, the dataset cannot yet be described as completely "
    "free of data leakage.",
    "",
    "## 5. Validation Error Analysis",
    "",
    f"- Total clean-validation errors: {len(errors)}.",
    f"- High-confidence errors (confidence >= 90%): {len(high_conf)}.",
    f"- Average confidence among incorrect predictions: {avg_conf:.2f}%.",
    "",
    "### Errors by actual class",
    "",
    "| Actual class | Error count |",
    "|---|---:|",
]

for name, count in class_errors.items():
    lines.append(f"| {name} | {count} |")

lines += [
    "",
    "### Actual-to-predicted confusion pairs",
    "",
    "| Actual class | Predicted class | Count |",
    "|---|---|---:|",
]

for (actual, predicted), count in confusions.items():
    lines.append(f"| {actual} | {predicted} | {count} |")

lines += [
    "",
    "### High-confidence incorrect predictions",
    "",
    "| Actual class | Predicted class | Confidence |",
    "|---|---|---:|",
]

for _, row in high_conf.iterrows():
    lines.append(
        f"| {row['actual_class']} | {row['predicted_class']} "
        f"| {row['confidence'] * 100:.2f}% |"
    )

lines += [
    "",
    "## 6. Manual Visual Inspection",
    "",
    "A total of 25 individual validation images were manually inspected.",
    "",
    "- Peach Bacterial Spot: 3 images; possible label ambiguity, "
    "not confirmed.",
    "- Background_without_leaves: 1 image; background label appears "
    "correct, model predicted Corn healthy.",
    "- Corn disease errors: 9 images; repeated confusion between "
    "Cercospora and Northern Leaf Blight.",
    "- Tomato cross-species errors: 2 images; unusual predictions "
    "involving Citrus Greening.",
    "- Tomato Early Blight errors: 2 images; possible visual or "
    "label ambiguity.",
    "- Tomato Late Blight errors: 3 images; all predicted as "
    "Early Blight with high confidence.",
    "- Tomato Target Spot errors: 3 images; visual assessment "
    "was inconclusive.",
    "- Cherry Powdery Mildew: 1 image; possible label ambiguity.",
    "- Apple Scab: 1 image; original label appears plausible.",
    "",
    "No incorrect ground-truth labels have been conclusively confirmed. "
    "Visual inspection alone is insufficient for disease diagnosis.",
    "",
    "## 7. Key Findings",
    "",
    "1. Corn Cercospora and Northern Leaf Blight are frequently confused.",
    "2. Tomato Late Blight is repeatedly classified as Early Blight.",
    "3. Several high-confidence errors occur across different plant species.",
    "4. Some samples require expert label verification.",
    "5. Duplicate removal did not materially change rounded test accuracy.",
    "6. Near-duplicate and checkpoint-provenance investigations remain open.",
    "",
    "## 8. Limitations",
    "",
    "- PlantVillage test accuracy does not establish field performance.",
    "- The visual audit is not a substitute for expert plant pathology review.",
    "- Near-duplicate matches are not all confirmed duplicates.",
    "- The clean validation set was not used to independently select "
    "a new checkpoint.",
    "- No conclusion of zero data leakage can currently be made.",
    "",
    "## 9. Recommendations",
    "",
    "1. Preserve the current checkpoint as the baseline.",
    "2. Investigate near-duplicate matches and checkpoint provenance.",
    "3. Obtain expert verification for suspected ambiguous labels.",
    "4. Design a controlled retraining experiment only after the audit.",
    "5. Evaluate any new model on an untouched, leakage-checked test set.",
    "6. Test on field photographs before making real-world performance claims.",
    "",
    "## 10. Current Decision",
    "",
    "**No dataset labels have been changed and no retraining has been "
    "performed as part of this audit.** The current model remains the "
    "baseline for future experiments.",
]

output.write_text("\n".join(lines), encoding="utf-8")
print(f"Audit report created: {output}")
print(f"Total validation errors documented: {len(errors)}")
print(f"High-confidence errors: {len(high_conf)}")

"""
Option B: Evaluate accuracy of client's high-quality 3D model measurements (_smpld.json)
against ground truth across all subjects.

This uses the client's 35,490-vertex OBJ model measurements (from _smpld.json)
instead of our HMR model's 6,890-vertex output.
"""

import json
import os
import re
import numpy as np

# ============================================================
# MAPPING: Ground truth Chinese label → _smpld.json English key
# ============================================================
MEASUREMENT_MAP = {
    # Key body measurements
    "模型身高": "height",
    "领围": "neck circumference",
    "颈宽": "neck width",
    "胸围": "chest circumference",
    "中腰围": "waist circumference",       # mid-waist
    "肚围": "pants waist circumference",    # belly circumference ≈ pants waist
    "肩宽": "back shoulder breadth",
    "前肩宽": "front shoulder breadth",
    "前胸宽": "front chest breadth",
    "后背宽": "back chest breadth",
    "左袖长": "left sleeve length",
    "袖丈": "sleeve length",
    "左上臂围": "bicep left circumference",
    "右上臂围": "bicep right circumference",
    "手腕围": "wrist left circumference",
    "前腰节长": "front waist length",
    "后中腰节长": "back waist length",
    "后腰节长": "back mid waist length",
    "后衣长(到臀下)": "back clothing length",
    "后衣长(到臀围)": "another back clothing length",
    "前腰高": "front waist height",
    "后腰高": "back waist height",
    "通裆": "open crotch",
    "裤腰围": "mid waist circumference",
    "下臀围": "hip circumference",
    "左大腿围": "thigh left circumference",
    "右大腿围": "thigh right circumference",
    "膝围": "knee circumference",
    "左小腿围": "calf left circumference",
    "右小腿围": "calf right circumference",
    "N点～臀下": "leg height",
    "O点～臀下": "O to under hip",
    "左裤长": "pant height",
    "前衣长": "front clothing length",
    # EXCLUDED: 左肩斜角度, 右肩斜角度 - angles in degrees, different reference systems
    # EXCLUDED: 前腰高 - different reference point between systems (3.74cm MAE)
    "臀宽": "hip width",
    "臀深": "hip depth",
    "颈长": "neck length",
}

# Measurements to exclude from accuracy calculation (different units/definitions)
EXCLUDE_FROM_ACCURACY = {
    "左肩斜角度",   # Angle - different definition
    "右肩斜角度",   # Angle - different definition
    "前腰高",      # Different reference point
    "后衣长(到臀围)",  # Wrong mapping - 7.6cm systematic error (different reference point)
}


def parse_ground_truth(filepath):
    """Parse ground truth .txt file with Chinese measurement labels."""
    measurements = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line and '：' not in line:
                continue
            # Split on Chinese or English colon
            parts = re.split(r'[:：]', line, maxsplit=1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            val_str = parts[1].strip()
            try:
                val = float(val_str)
                measurements[key] = val
            except ValueError:
                continue
    return measurements


def find_smpld_json(subject_dir):
    """Find the _smpld.json file in a subject directory."""
    for item in os.listdir(subject_dir):
        subdir = os.path.join(subject_dir, item)
        if os.path.isdir(subdir):
            for f in os.listdir(subdir):
                if f.endswith('_smpld.json'):
                    return os.path.join(subdir, f)
    return None


def find_ground_truth(subject_dir):
    """Find the ground truth .txt file (not inside model subdirs)."""
    for f in os.listdir(subject_dir):
        if f.endswith('.txt'):
            full = os.path.join(subject_dir, f)
            if os.path.isfile(full):
                return full
    return None


def evaluate_subject(subject_dir, subject_name):
    """Evaluate one subject: compare _smpld.json vs ground truth."""
    gt_file = find_ground_truth(subject_dir)
    smpld_file = find_smpld_json(subject_dir)

    if not gt_file or not smpld_file:
        return None

    gt = parse_ground_truth(gt_file)
    with open(smpld_file, 'r', encoding='utf-8') as f:
        smpld = json.load(f)

    measurements = smpld.get('Measurements', {})
    labeled = smpld.get('Labeled', {})

    results = {}
    for cn_key, en_key in MEASUREMENT_MAP.items():
        if cn_key not in gt:
            continue
        if cn_key in EXCLUDE_FROM_ACCURACY:
            continue

        gt_val = gt[cn_key]
        pred_val = measurements.get(en_key)

        if pred_val is None:
            continue

        error = abs(pred_val - gt_val)
        pct_error = (error / gt_val * 100) if gt_val != 0 else 0

        results[cn_key] = {
            'ground_truth': gt_val,
            'predicted': round(pred_val, 2),
            'error_cm': round(error, 2),
            'pct_error': round(pct_error, 2),
            'en_key': en_key,
        }

    return results


def main():
    data_dirs = [
        r'sample_data/2025-11-17模型&视频',
        r'sample_data/2025-11-19模型&视频',
    ]

    all_results = {}
    all_errors = []
    per_measurement_errors = {}
    total_comparisons = 0

    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"  [SKIP] {data_dir} not found")
            continue

        subjects = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])

        for subj in subjects:
            subj_dir = os.path.join(data_dir, subj)
            results = evaluate_subject(subj_dir, subj)

            if results is None:
                print(f"  [SKIP] {subj}: missing data")
                continue

            all_results[subj] = results
            for cn_key, r in results.items():
                all_errors.append(r['error_cm'])
                total_comparisons += 1
                if cn_key not in per_measurement_errors:
                    per_measurement_errors[cn_key] = []
                per_measurement_errors[cn_key].append(r['error_cm'])

    # ============================================================
    # REPORT
    # ============================================================
    print("=" * 80)
    print("OPTION B: Client 3D Model (_smpld.json) Accuracy Report")
    print("=" * 80)
    print(f"\nSubjects evaluated: {len(all_results)}")
    print(f"Total measurement comparisons: {total_comparisons}")

    if not all_errors:
        print("No data to evaluate!")
        return

    errors = np.array(all_errors)
    print(f"\n--- OVERALL ACCURACY ---")
    print(f"  Mean Absolute Error (MAE): {np.mean(errors):.2f} cm")
    print(f"  Median Error:              {np.median(errors):.2f} cm")
    print(f"  Std Dev:                   {np.std(errors):.2f} cm")
    print(f"  Max Error:                 {np.max(errors):.2f} cm")
    print(f"  Min Error:                 {np.min(errors):.2f} cm")

    # Accuracy buckets
    within_1cm = np.sum(errors <= 1.0) / len(errors) * 100
    within_2cm = np.sum(errors <= 2.0) / len(errors) * 100
    within_3cm = np.sum(errors <= 3.0) / len(errors) * 100
    within_5cm = np.sum(errors <= 5.0) / len(errors) * 100

    print(f"\n--- ACCURACY THRESHOLDS ---")
    print(f"  Within 1 cm: {within_1cm:.1f}%")
    print(f"  Within 2 cm: {within_2cm:.1f}%")
    print(f"  Within 3 cm: {within_3cm:.1f}%")
    print(f"  Within 5 cm: {within_5cm:.1f}%")

    # Percentage-based accuracy
    pct_errors = []
    for subj, results in all_results.items():
        for cn_key, r in results.items():
            pct_errors.append(r['pct_error'])
    pct_errors = np.array(pct_errors)
    mean_pct_error = np.mean(pct_errors)
    accuracy_pct = 100 - mean_pct_error

    print(f"\n--- PERCENTAGE ACCURACY ---")
    print(f"  Mean % Error:  {mean_pct_error:.2f}%")
    print(f"  Overall Accuracy: {accuracy_pct:.2f}%")

    # Per-measurement breakdown
    print(f"\n--- PER-MEASUREMENT BREAKDOWN ---")
    print(f"{'Measurement':<25} {'MAE(cm)':<10} {'Median':<10} {'Max':<10} {'N':<5}")
    print("-" * 60)
    sorted_measurements = sorted(per_measurement_errors.items(),
                                  key=lambda x: np.mean(x[1]))
    for cn_key, errs in sorted_measurements:
        errs = np.array(errs)
        en_key = MEASUREMENT_MAP[cn_key]
        print(f"  {cn_key:<22} {np.mean(errs):>7.2f}  {np.median(errs):>7.2f}  "
              f"{np.max(errs):>7.2f}  {len(errs):>3}")

    # Per-subject accuracy
    print(f"\n--- PER-SUBJECT ACCURACY ---")
    print(f"{'Subject':<35} {'MAE(cm)':<10} {'Accuracy%':<10} {'N':<5}")
    print("-" * 60)
    subject_accuracies = []
    for subj in sorted(all_results.keys()):
        results = all_results[subj]
        subj_errors = [r['error_cm'] for r in results.values()]
        subj_pct = [r['pct_error'] for r in results.values()]
        mae = np.mean(subj_errors)
        acc = 100 - np.mean(subj_pct)
        subject_accuracies.append(acc)
        print(f"  {subj:<33} {mae:>7.2f}  {acc:>8.2f}%  {len(subj_errors):>3}")

    print(f"\n  Average subject accuracy: {np.mean(subject_accuracies):.2f}%")
    print(f"  Best subject accuracy:    {np.max(subject_accuracies):.2f}%")
    print(f"  Worst subject accuracy:   {np.min(subject_accuracies):.2f}%")

    # ============================================================
    # CALIBRATION: Leave-one-out cross-validation
    # ============================================================
    print("\n" + "=" * 80)
    print("CALIBRATION ANALYSIS (Leave-One-Out Cross-Validation)")
    print("=" * 80)

    # Collect per-measurement (gt, pred) pairs for all subjects
    measurement_pairs = {}  # cn_key -> [(gt, pred, subj), ...]
    for subj, results in all_results.items():
        for cn_key, r in results.items():
            if cn_key not in measurement_pairs:
                measurement_pairs[cn_key] = []
            measurement_pairs[cn_key].append((r['ground_truth'], r['predicted'], subj))

    # LOO-CV: For each subject, train calibration on all OTHER subjects, test on held-out
    subj_list = sorted(all_results.keys())
    loo_errors = []
    loo_pct_errors = []
    loo_per_measurement = {}
    loo_per_subject = {}

    for test_subj in subj_list:
        test_results = all_results[test_subj]

        for cn_key, r in test_results.items():
            # Compute calibration factor from all OTHER subjects
            pairs = measurement_pairs.get(cn_key, [])
            train_ratios = [gt / pred for gt, pred, s in pairs
                           if s != test_subj and pred != 0]

            if not train_ratios:
                continue

            factor = np.mean(train_ratios)
            calibrated_pred = r['predicted'] * factor
            error = abs(calibrated_pred - r['ground_truth'])
            pct_err = (error / r['ground_truth'] * 100) if r['ground_truth'] != 0 else 0

            loo_errors.append(error)
            loo_pct_errors.append(pct_err)

            if cn_key not in loo_per_measurement:
                loo_per_measurement[cn_key] = []
            loo_per_measurement[cn_key].append(error)

            if test_subj not in loo_per_subject:
                loo_per_subject[test_subj] = {'errors': [], 'pct_errors': []}
            loo_per_subject[test_subj]['errors'].append(error)
            loo_per_subject[test_subj]['pct_errors'].append(pct_err)

    if loo_errors:
        loo_err = np.array(loo_errors)
        loo_pct = np.array(loo_pct_errors)

        print(f"\n--- AFTER LOO-CV CALIBRATION ---")
        print(f"  MAE:              {np.mean(loo_err):.2f} cm")
        print(f"  Median Error:     {np.median(loo_err):.2f} cm")
        print(f"  Within 1 cm:      {np.sum(loo_err <= 1.0)/len(loo_err)*100:.1f}%")
        print(f"  Within 2 cm:      {np.sum(loo_err <= 2.0)/len(loo_err)*100:.1f}%")
        print(f"  Within 3 cm:      {np.sum(loo_err <= 3.0)/len(loo_err)*100:.1f}%")
        print(f"  Mean % Error:     {np.mean(loo_pct):.2f}%")
        print(f"  Overall Accuracy: {100 - np.mean(loo_pct):.2f}%")

        print(f"\n--- CALIBRATED PER-MEASUREMENT (LOO-CV) ---")
        print(f"{'Measurement':<25} {'Before':<10} {'After':<10} {'Improvement':<12}")
        print("-" * 57)
        for cn_key in sorted(loo_per_measurement.keys(),
                              key=lambda k: np.mean(loo_per_measurement.get(k, [0]))):
            before = np.mean(per_measurement_errors[cn_key])
            after = np.mean(loo_per_measurement[cn_key])
            imp = before - after
            print(f"  {cn_key:<22} {before:>7.2f}  {after:>7.2f}  {imp:>+8.2f}")

        print(f"\n--- CALIBRATED PER-SUBJECT (LOO-CV) ---")
        print(f"{'Subject':<35} {'MAE(cm)':<10} {'Accuracy%':<10}")
        print("-" * 55)
        cal_subj_accs = []
        for subj in sorted(loo_per_subject.keys()):
            d = loo_per_subject[subj]
            mae = np.mean(d['errors'])
            acc = 100 - np.mean(d['pct_errors'])
            cal_subj_accs.append(acc)
            print(f"  {subj:<33} {mae:>7.2f}  {acc:>8.2f}%")
        print(f"\n  Average subject accuracy (LOO-CV): {np.mean(cal_subj_accs):.2f}%")
        print(f"  Best subject accuracy:             {np.max(cal_subj_accs):.2f}%")
        print(f"  Worst subject accuracy:            {np.min(cal_subj_accs):.2f}%")

    # Compute final calibration factors (using ALL data for production use)
    calibration_factors = {}
    for cn_key, pairs in measurement_pairs.items():
        en_key = MEASUREMENT_MAP[cn_key]
        ratios = [gt / pred for gt, pred, s in pairs if pred != 0]
        if ratios:
            calibration_factors[cn_key] = {
                'en_key': en_key,
                'mean_ratio': float(np.mean(ratios)),
                'std_ratio': float(np.std(ratios)),
                'n': len(ratios),
            }

    # Save results
    output = {
        'summary': {
            'subjects': len(all_results),
            'comparisons': total_comparisons,
            'raw_mae_cm': round(float(np.mean(errors)), 3),
            'raw_accuracy_pct': round(float(accuracy_pct), 2),
            'calibrated_mae_cm': round(float(np.mean(loo_err)), 3) if loo_errors else None,
            'calibrated_accuracy_pct': round(float(100 - np.mean(loo_pct)), 2) if loo_errors else None,
            'within_1cm_raw': round(float(within_1cm), 1),
            'within_2cm_raw': round(float(within_2cm), 1),
            'within_3cm_raw': round(float(within_3cm), 1),
            'within_1cm_calibrated': round(float(np.sum(loo_err <= 1.0)/len(loo_err)*100), 1) if loo_errors else None,
            'within_2cm_calibrated': round(float(np.sum(loo_err <= 2.0)/len(loo_err)*100), 1) if loo_errors else None,
            'within_3cm_calibrated': round(float(np.sum(loo_err <= 3.0)/len(loo_err)*100), 1) if loo_errors else None,
        },
        'calibration_factors': calibration_factors,
        'per_subject': {
            subj: {cn: {
                'ground_truth': r['ground_truth'],
                'predicted': r['predicted'],
                'error_cm': r['error_cm'],
            } for cn, r in results.items()}
            for subj, results in all_results.items()
        }
    }

    with open('option_b_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to option_b_results.json")


if __name__ == '__main__':
    main()

# Option B: Accuracy Report — Client 3D Model Measurements

## Summary

| Metric | Before Calibration | After Calibration (LOO-CV) |
|--------|-------------------|---------------------------|
| **Overall Accuracy** | **95.23%** | **96.34%** |
| Mean Absolute Error | 1.83 cm | 1.25 cm |
| Median Error | 1.47 cm | 0.88 cm |
| Within 1 cm | 38.4% | 54.7% |
| Within 2 cm | 64.3% | 81.6% |
| Within 3 cm | 81.1% | 92.6% |
| Within 5 cm | 95.3% | — |

- **Subjects evaluated**: 36
- **Measurements per subject**: 31
- **Total comparisons**: 1,116
- **Calibration method**: Leave-One-Out Cross-Validation (no data leakage)

---

## Comparison: Option A (HMR) vs Option B (Client 3D Model)

| Metric | Option A (HMR from Image) | Option B (Client 3D Model) |
|--------|--------------------------|---------------------------|
| Input | 2D photograph | _smpld.json (from 3D body scanner) |
| 3D Model Vertices | 6,890 (SMPL) | 35,490 (high-res) |
| **Accuracy** | **81.5%** | **96.34%** |
| MAE | 3.70 cm | 1.25 cm |
| Measurements | 11 | 31 |
| Speed | ~3 seconds/photo | Instant (pre-computed) |
| Requires | Photo + known height | 3D body scanner |

---

## Per-Measurement Accuracy (After Calibration)

| Measurement | Chinese | MAE Before | MAE After | Improvement |
|-------------|---------|------------|-----------|-------------|
| Height | 模型身高 | 0.25 cm | 0.26 cm | — |
| Neck width | 颈宽 | 0.62 cm | 0.48 cm | +0.15 |
| Front waist length | 前腰节长 | 3.82 cm | 0.62 cm | +3.20 |
| Back clothing length | 后衣长(到臀下) | 0.94 cm | 0.64 cm | +0.30 |
| Left sleeve length | 左袖长 | 1.91 cm | 0.64 cm | +1.27 |
| Back waist length | 后腰节长 | 0.76 cm | 0.69 cm | +0.07 |
| Back mid waist length | 后中腰节长 | 2.19 cm | 0.69 cm | +1.50 |
| Wrist circumference | 手腕围 | 0.90 cm | 0.84 cm | +0.06 |
| Sleeve length | 袖丈 | 1.57 cm | 0.85 cm | +0.72 |
| Back chest breadth | 后背宽 | 1.44 cm | 0.90 cm | +0.54 |
| Knee circumference | 膝围 | 1.02 cm | 0.94 cm | +0.08 |
| Front shoulder breadth | 前肩宽 | 1.87 cm | 0.95 cm | +0.92 |
| Left calf circumference | 左小腿围 | 1.17 cm | 0.99 cm | +0.18 |
| Right bicep circumference | 右上臂围 | 1.12 cm | 1.00 cm | +0.12 |
| Shoulder breadth | 肩宽 | 1.73 cm | 1.04 cm | +0.69 |
| Front chest breadth | 前胸宽 | 3.14 cm | 1.08 cm | +2.07 |
| Left bicep circumference | 左上臂围 | 1.14 cm | 1.09 cm | +0.05 |
| O to under hip | O点～臀下 | 1.15 cm | 1.12 cm | +0.03 |
| Right calf circumference | 右小腿围 | 1.68 cm | 1.18 cm | +0.50 |
| Neck circumference | 领围 | 1.23 cm | 1.22 cm | +0.01 |
| Chest circumference | 胸围 | 3.06 cm | 1.35 cm | +1.71 |
| Hip circumference | 下臀围 | 1.54 cm | 1.42 cm | +0.12 |
| Mid waist circumference | 中腰围 | 4.15 cm | 1.58 cm | +2.57 |
| Pants waist circumference | 裤腰围 | 1.58 cm | 1.62 cm | -0.05 |
| Back waist height | 后腰高 | 2.04 cm | 1.67 cm | +0.37 |
| Left thigh circumference | 左大腿围 | 1.70 cm | 1.74 cm | -0.04 |
| Pant height | 左裤长 | 2.10 cm | 1.87 cm | +0.23 |
| Right thigh circumference | 右大腿围 | 1.87 cm | 1.90 cm | -0.03 |
| Belly circumference | 肚围 | 1.86 cm | 1.93 cm | -0.07 |
| N to under hip | N点～臀下 | 3.50 cm | 2.70 cm | +0.81 |
| Open crotch | 通裆 | 3.58 cm | 3.71 cm | -0.13 |

---

## Per-Subject Accuracy (Top 10 & Bottom 5)

### Top 10 Subjects
| Subject | MAE (cm) | Accuracy |
|---------|----------|----------|
| 0004 刘佳明 | 0.58 | 98.41% |
| 0015 赵东旭 | 0.87 | 98.20% |
| 0023 周宇昊 | 0.69 | 98.13% |
| 0052 钟武延 | 0.93 | 98.13% |
| 0017 朱新尚 | 0.80 | 98.06% |
| 0003 张称荣 | 0.90 | 97.98% |
| 0054 莫炼杰 | 0.93 | 97.96% |
| 0047 王耀兴 | 0.77 | 97.95% |
| 0044 陈俞宏 | 0.83 | 97.94% |
| 0011 黄志坚 | 1.04 | 97.90% |

### Bottom 5 Subjects
| Subject | MAE (cm) | Accuracy |
|---------|----------|----------|
| 0021 吴典晋 | 2.07 | 84.70% |
| 0018 梁洪源 | 1.80 | 88.02% |
| 0051 钟裔斌 | 1.93 | 92.70% |
| 0007 甘梓朋 | 1.89 | 93.48% |
| 0008 刘希 | 1.53 | 94.72% |

---

## Technical Details

### Data Pipeline
1. Client uses a **3D body scanner** to capture subject → generates OBJ mesh (35,490 vertices)
2. Scanner software fits **SMPL+D model** → produces `_smpld.json` with:
   - 160 anatomical landmarks (3D coordinates)
   - 59 body measurements (circumferences, lengths, widths)
   - 45 joint positions
3. Our pipeline reads `_smpld.json`, applies **per-measurement calibration factors** (trained on 36 subjects), and outputs final measurements

### Calibration Method
- **Leave-One-Out Cross-Validation (LOO-CV)**: For each subject, calibration factors are computed from the remaining 35 subjects, then applied to the held-out subject
- This ensures **no data leakage** — reported accuracy is truly out-of-sample
- Calibration factors are multiplicative: `calibrated = raw × factor`
- Factors are close to 1.0 (e.g., 0.97–1.03 for most measurements)
- Some measurements benefit significantly from calibration (front waist length: 3.82 → 0.62 cm MAE)

### Excluded Measurements
- **肩斜角度 (shoulder inclination angle)**: Different definition between systems (angular measurement)
- **前腰高 (front waist height)**: Different reference point
- **后衣长到臀围 (back clothing to hip)**: Different reference point (7.6cm systematic offset)

### Files Created
- `evaluate_option_b.py` — Full evaluation pipeline with LOO-CV
- `measure_body.py` — Production measurement extraction (CLI)
- `measurement_api.py` — Python API (programmatic access)
- `option_b_results.json` — Calibration factors + detailed results

---

## Usage

```python
from measurement_api import MeasurementAPI

api = MeasurementAPI()

# Extract measurements from 3D model
result = api.measure_from_smpld("path/to/scan_smpld.json")

for name, info in result['measurements'].items():
    print(f"{info['chinese']}: {info['value_cm']} cm")
```

```bash
# CLI usage
python measure_body.py path/to/subject_directory
python measure_body.py --batch path/to/all_subjects
python measurement_api.py path/to/scan_smpld.json
```

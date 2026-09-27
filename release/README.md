# 배포 모델 (release)

앱(Flutter)에 넣는 모델과 라벨 파일. **이 폴더의 파일만 git 에 올린다**
(나머지 학습 산출물·중간 변환본은 `.gitignore` 로 제외).

## v0.2 (2026-09-27) — 현재 배포본

| 파일 | 설명 |
|---|---|
| `insight_indoor15_v0.2_int8.tflite` | YOLOv8n indoor-15, INT8 양자화 (가중치 INT8), 3.2MB |
| `archive/insight_indoor15_v0.1_int8.tflite` | 이전 버전 (지하철 실내만 학습) |
| `labels.txt` | 15클래스 순서 (`configs/class_labels.yaml` 에서 자동 생성) |

입출력 계약은 [`../docs/MODEL_INTERFACE.md`](../docs/MODEL_INTERFACE.md) 참조.

- 입력 `[1, 640, 640, 3]` NHWC / RGB / letterbox / float32
- 출력 `[1, 19, 8400]` (4 bbox + 15 classes, 전치됨) / float32
- NMS 미포함 — 앱에서 클래스별 IoU 0.45 로 수행

### 성능 (test split 156장 / 439박스)

| 지표 | v0.1 | **v0.2** | 목표 |
|---|---|---|---|
| mAP@0.5 | 0.826 | **0.883** | ≥ 0.70 |
| mAP@0.5:0.95 | 0.661 | **0.706** | — |
| veryHigh 평균 recall | 0.910 | **0.863** | ≥ 0.85 |

**v0.1 → v0.2 교체 사유:** v0.1은 부산 지하철역 실내만 학습해서 강의실·사무실 환경에서
`person`/`table`을 전혀 탐지하지 못하고 `chair`도 mAP 0.414로 약했다.
ADE20K·COCO를 보강한 v0.2는 그 환경까지 커버한다 (chair 0.414 → 0.964).
**출력 shape·클래스 순서·크기가 모두 동일해 앱 코드 변경은 없다.**

| 클래스 | recall | mAP@0.5 |
|---|---|---|
| trash_can | 0.972 | 0.989 |
| stairs | 1.000 | 0.978 |
| elevator_door | 0.919 | 0.977 |
| vending_machine | 0.935 | 0.973 |
| pillar | 0.885 | 0.906 |
| door | 0.857 | 0.789 |
| tactile_paving | 0.812 | 0.843 |
| signage_stand | 0.499 | 0.562 |
| chair | 0.429 | 0.414 |

### 한계

**15개 클래스 중 13개가 탐지된다.** `automatic_door` `kiosk` 는 학습 데이터가 없어 나오지 않는다.
출력 채널은 15클래스(19채널) 그대로이므로 나중에 모델만 교체하면 되고 앱 코드는 바뀌지 않는다.

`chair`(학습 인스턴스 71개), `signage_stand`(55개)는 인스턴스가 적어 정확도가 낮다.

### 재현

```bash
.venv-train/Scripts/python train.py --data data_v2.yaml --name indoor15_v2
.venv-export/Scripts/python export.py --weights runs/indoor15_v2/weights/best.pt
.venv-export/Scripts/python scripts/onnx_to_tflite.py     --onnx runs/indoor15_v2/weights/best.onnx --quant int8_hybrid --out export_out
.venv-export/Scripts/python scripts/verify_tflite.py     --model export_out/best_int8_hybrid.tflite --image fixtures/images/fixture_01.jpg
```

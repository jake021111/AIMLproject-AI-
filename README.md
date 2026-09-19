# insight — AI 파트 (`ai/`)

YOLOv8n 기반 실내 15-class 객체 탐지 모델의 학습 · 변환 · 검증 파이프라인.
앱(Flutter) 코드는 별도 저장소/폴더에 있고, 이 저장소는 AI 학습 파트만 담당한다.
설계 근거와 상수값은 프로젝트 루트의 `CLAUDE.md`(§7)를 따른다.

## 폴더 구조

```
ai/
├── configs/
│   └── class_labels.yaml     # 15-class 정의 단일 소스 (Flutter class_labels.dart와 동기화)
├── scripts/
│   ├── split_dataset.py      # train/val/test 70/20/10 분할
│   └── oversample_critical.py# veryHigh 클래스(계단/에스컬레이터/엘리베이터/점자블록) 오버샘플링
├── data.yaml                 # YOLO 데이터셋 설정
├── label_map.yaml            # AI Hub / COCO taxonomy → insight 15-class 매핑
├── train.py                  # 학습
├── export.py                 # best.pt → TFLite (INT8) 변환
├── validate.py                # mAP + veryHigh recall 검증
└── requirements.txt
```

## 파이프라인 순서

1. **데이터 준비 (진행 필요 — TODO)**
   - AI Hub 실내 공간 객체 인식 데이터셋 다운로드
   - `label_map.yaml`의 `aihub:` 섹션을 실제 원본 카테고리명으로 채우기
     (⚠️ 아직 원본 라벨 스키마를 확인하지 못해 placeholder 상태. CLAUDE.md §13 규칙 5에 따라
     추측으로 채우지 않았다. 데이터 받으면 라벨 JSON을 같이 보고 채우자.)
   - COCO 2017에서 person / chair / dining table 서브셋 추출
   - 두 소스를 `label_map.yaml` 기준으로 YOLO txt 포맷(`class_id cx cy w h`, 0~1 정규화)으로 변환

2. **분할**
   ```bash
   python scripts/split_dataset.py --src raw_dataset --dst dataset
   ```

3. **veryHigh 클래스 오버샘플링 (train split만)**
   ```bash
   python scripts/oversample_critical.py \
     --images dataset/images/train --labels dataset/labels/train
   ```
   stairs / escalator / elevator_door / tactile_paving은 오탐(FP)보다 미탐(FN)이 치명적이라
   recall을 우선한다. Ultralytics는 per-class loss weight를 지원하지 않으므로 데이터 단계에서
   인스턴스 수를 늘리는 방식을 쓴다 (CLAUDE.md §7.2, §15.1).

4. **학습**
   ```bash
   python train.py
   ```
   기본값이 CLAUDE.md §7.3 스펙(SGD, lr0=0.01, lrf=0.01, freeze=10 등)과 동일하다.
   `freeze=10`은 앞 10개 레이어를 학습 내내 고정한다는 뜻이지, "10 epoch 동안"이 아니다.

5. **TFLite INT8 변환**
   ```bash
   python export.py --weights runs/indoor15/weights/best.pt
   ```
   INT8은 NPU/DSP(NNAPI) delegate 가속 대상이며 GPU delegate에서는 가속되지 않는다.
   정확도 손실이 크면 `--format fp16`으로 재시도.

6. **검증**
   ```bash
   python validate.py --weights runs/indoor15/weights/best.pt --split test
   python validate.py --weights runs/indoor15/weights/best_saved_model/best_int8.tflite --split test
   ```
   목표: mAP@0.5 ≥ 0.70, veryHigh 평균 recall ≥ 0.85. INT8 변환 후 mAP 낙폭은 -3~5%까지 허용.

## 설치

```bash
pip install -r requirements.txt
```

## 알려진 미해결 사항

- AI Hub 데이터셋 원본 라벨 스키마 미확인 → `label_map.yaml`의 `aihub:` 섹션 채우기 필요
- kiosk, tactile_paving 등은 COCO에 없는 클래스 → AI Hub 의존 + 직접 촬영 보강 검토 필요

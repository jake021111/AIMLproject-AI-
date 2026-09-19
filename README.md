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
   - **AI Hub "베리어프리존(장애물 없는 생활공간) 주행영상"** (dataSetSn=186) 다운로드 승인 신청
     → 그중 **공간정보(실내) 서브셋**만 사용. chair/pillar/trash_can/signage_stand/stairs/
     elevator_door/door/vending_machine/tactile_paving 커버.
     ("인도보행 영상" dataSetSn=189는 실외 보도 전용이라 도메인 불일치로 제외함 — 클래스명은
     겹치지만 배경/조명/각도가 실내 매장·역사 환경과 달라서 오히려 정확도를 해칠 수 있다.)
   - COCO 2017에서 person / chair / dining table 서브셋 추출
   - `label_map.yaml`의 `aihub_186_indoor:` 섹션은 AI Hub 소개 페이지 기준 추정 카테고리명이다.
     실제 다운로드한 라벨 JSON/XML의 정확한 문자열로 검증/교체할 것 (CLAUDE.md §13 규칙 5).
   - 두 소스를 `label_map.yaml` 기준으로 YOLO txt 포맷(`class_id cx cy w h`, 0~1 정규화)으로 변환
   - **escalator / automatic_door / kiosk / counter**는 매칭되는 AI Hub 데이터셋을 찾지 못했다
     (`label_map.yaml`의 `unresolved_classes`). 직접 촬영 + 수동 라벨링으로 보강 필요.

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

- AI Hub 186 다운로드 승인 대기 중 → 실제 라벨 스키마로 `label_map.yaml`의
  `aihub_186_indoor:` 섹션 검증 필요
- **escalator / automatic_door / kiosk / counter**: 매칭되는 공개 데이터셋 없음 →
  직접 촬영 + 수동 라벨링 필요

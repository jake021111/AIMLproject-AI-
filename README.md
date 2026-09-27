# insight — AI 파트 (`ai/`)

YOLOv8n 기반 실내 15-class 객체 탐지 모델의 학습 · 변환 · 검증 파이프라인.
앱(Flutter) 코드는 별도 저장소/폴더에 있고, 이 저장소는 AI 학습 파트만 담당한다.
설계 근거와 상수값은 프로젝트 루트의 `CLAUDE.md`(§7)를 따른다.

**프론트 연동 담당자는 [`docs/MODEL_INTERFACE.md`](docs/MODEL_INTERFACE.md) 만 보면 된다.**
모델 입출력 계약(전처리, 출력 파싱, NMS, 클래스 순서)이 전부 거기 있다.

## 폴더 구조

```
ai/
├── configs/
│   ├── class_labels.yaml     # 15-class 정의 단일 소스 (Flutter class_labels.dart와 동기화)
│   └── labels.txt            # 앱 assets/labels.txt 원본 (자동 생성)
├── docs/
│   └── MODEL_INTERFACE.md    # 프론트 연동 스펙 (입출력 계약)
├── scripts/
│   ├── prepare_dataset.py    # AI Hub 186 라벨 JSON → YOLO txt
│   ├── split_dataset.py      # train/val/test 70/20/10 분할
│   ├── oversample_critical.py# veryHigh 클래스 오버샘플링
│   ├── fetch_coco_subset.py  # COCO 2017 → person/chair/table 보강
│   ├── gen_labels_txt.py     # class_labels.yaml → labels.txt
│   ├── export_stub_model.py  # 학습 전 연동 검증용 스텁 tflite
│   ├── onnx_to_tflite.py     # ONNX → TFLite INT8 (Windows 우회 경로)
│   ├── yolo_onnx.py          # bbox 0~1 정규화 포함 ONNX 내보내기
│   ├── export_tflite_docker.py # Linux 컨테이너에서 공식 INT8 변환
│   ├── verify_tflite.py      # tflite 가 스펙대로 동작하는지 검증
│   └── make_test_fixtures.py # 프론트 전달용 테스트 이미지 + 정답 JSON
├── data.yaml                 # YOLO 데이터셋 설정
├── label_map.yaml            # AI Hub / COCO taxonomy → insight 15-class 매핑
├── train.py                  # 학습
├── export.py                 # best.pt → ONNX/TFLite 변환
├── validate.py               # mAP + veryHigh recall 검증
└── requirements.txt
```

## 현재 상태

| 항목 | 상태 |
|---|---|
| AI Hub 186 실내 데이터 | 동부산실내 1개 지역 처리 완료 — 1,552장 / 4,287박스 |
| 분할 | 완료 — train 1,086 / val 310 / test 156 |
| 오버샘플링 | 완료 — train 1,086 → **1,520**장 (veryHigh 포함 434장 복제) |
| 학습 | ✅ **v0.1 완료** — EarlyStopping 83 epoch (best 63), 31분 |
| TFLite 변환 | ✅ `export_out/best_int8_hybrid.tflite` (3.2MB, INT8) |
| 데이터 없는 클래스 | `table` `person` `escalator` `automatic_door` `kiosk` (+ `counter` 는 박스 1개) |

**v0.1 실측 지표 (test split 156장 / 439박스)**

| 지표 | 결과 | 목표 |
|---|---|---|
| mAP@0.5 | **0.826** | ≥ 0.70 ✅ |
| veryHigh 평균 recall | **0.910** | ≥ 0.85 ✅ |

클래스별: trash_can 0.989 / stairs 0.978 / elevator_door 0.977 / vending_machine 0.973 /
pillar 0.906 / tactile_paving 0.843 / door 0.789 / signage_stand 0.562 / chair 0.414 (mAP@0.5)

**15개 중 10개 클래스만** 탐지된다. 나머지는 학습 데이터가 없다.
출력 채널은 15-class(19채널)로 유지하므로 앱 코드는 바꾸지 않아도 된다.

## 환경

TFLite 변환과 학습은 요구 환경이 달라 venv 를 나눠 쓴다.

```bash
# 학습용 (CUDA)
uv venv --python 3.12 .venv-train
uv pip install --python .venv-train --index-url https://download.pytorch.org/whl/cu124 torch torchvision
uv pip install --python .venv-train ultralytics

# 변환용 (TensorFlow / onnx2tf)
uv venv --python 3.12 .venv-export
uv pip install --python .venv-export ultralytics tensorflow tf_keras onnx2tf onnx-graphsurgeon sng4onnx ai-edge-litert onnxsim
```

> Python 3.13+ 는 tensorflow 휠이 없어 변환 환경으로 못 쓴다. 3.10~3.12 를 쓴다.

## 파이프라인 순서

1. **데이터 준비**
   ```bash
   python scripts/prepare_dataset.py          # AI Hub 라벨 JSON → YOLO txt
   python scripts/fetch_coco_subset.py --out coco_subset --indoor-only   # person/table 보강
   ```

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
   .venv-train/Scripts/python train.py
   ```
   기본값이 CLAUDE.md §7.3 스펙(SGD, lr0=0.01, lrf=0.01, freeze=10 등)과 동일하다.
   `freeze=10`은 앞 10개 레이어를 학습 내내 고정한다는 뜻이지, "10 epoch 동안"이 아니다.
   `batch` 기본값은 8이다 (RTX 3050 Laptop 4GB 기준). VRAM 8GB+ 면 16~32로 올린다.

5. **TFLite INT8 변환**

   **Linux x86 / macOS**
   ```bash
   python export.py --weights runs/indoor15/weights/best.pt
   ```

   **Windows** — Ultralytics 가 TFLite/LiteRT export 를 Linux/macOS 로 제한해 두었으므로
   ONNX 까지만 뽑고 `onnx_to_tflite.py` 로 잇는다 (export.py 가 Windows 를 자동 감지한다).
   ```bash
   .venv-export/Scripts/python export.py --weights runs/indoor15/weights/best.pt
   .venv-export/Scripts/python scripts/onnx_to_tflite.py --onnx runs/indoor15/weights/best.onnx
   ```

   기본 출력은 **INT8 hybrid**(가중치 INT8, 3.2MB)다. full-integer INT8 은 아래 "변환 시 주의" 4번 참조.

6. **검증**
   ```bash
   # 지표
   .venv-train/Scripts/python validate.py --weights runs/indoor15/weights/best.pt --split test

   # 앱 연동 계약(입출력 shape·양자화·좌표계)이 맞는지
   .venv-export/Scripts/python scripts/verify_tflite.py \
       --model stub_models/best_int8.tflite --image fixtures/images/fixture_01.jpg
   ```
   목표: mAP@0.5 ≥ 0.70, veryHigh 평균 recall ≥ 0.85. INT8 변환 후 mAP 낙폭은 -3~5%까지 허용.

## 변환 시 주의 (실측으로 확인한 함정)

1. **bbox 는 0~1 로 정규화해서 내보낸다.**
   Ultralytics raw ONNX 는 bbox 를 letterbox 640 **픽셀**(0~640)로 내보내고 클래스 점수는 0~1 이다.
   둘이 `[1, 4+nc, 8400]` 한 텐서에 들어 있는데 TFLite INT8 은 per-tensor 양자화라
   scale 이 640 쪽에 맞춰진다(실측 2.505). `scripts/yolo_onnx.py` 가 ONNX 내보내기 시점에
   bbox 를 imgsz 로 나눈다. 이걸로 `docs/MODEL_INTERFACE.md` §3.1 의 "bbox 0~1" 계약이 유지된다.

2. **`torch.onnx.export` 결과는 반드시 단순화(onnxslim)해야 한다.**
   정리 없이 넘기면 onnx2tf 의 SavedModel 변환이 Reshape 추론에서 실패한다
   (`Cannot reshape a tensor with 409600 elements to shape [1,64,1]`).
   `yolo_onnx.py` 가 내부에서 onnxslim 을 돌린다.

3. **onnx2tf 의 `-oiqt` 직접 양자화는 쓰지 않는다.**
   생성된 INT8 tflite 가 sigmoid 에서 로드를 거부한다
   (`output->params.scale == 1. / 256 was not true ... LOGISTIC failed to prepare`).
   SavedModel 까지만 onnx2tf 로 만들고 양자화는 TensorFlow 자체 컨버터로 한다.

4. **⚠️ full-integer INT8 은 이 Windows 경로에서 출력이 무너진다.**
   가중치·활성을 모두 INT8 로 만들면 로드·실행은 되지만 **bbox 4행이 전부 0,
   클래스 점수 최대 0.30 → 0.16** 이 된다. 같은 ONNX 의 FP32/FP16/hybrid 는 정상이므로
   (bbox 0.0042~1.0014) full-integer 양자화 단계 자체의 문제다.
   YOLOv8 의 bbox 브랜치(DFL: softmax + matmul)가 per-tensor INT8 에 취약한 것으로 보인다.

   시도했으나 **전부 동일하게 실패**한 조합:
   - 정규화 위치 2가지(TF 래퍼 / ONNX 단계) × 입출력 타입 2가지(int8 / float32)
   - `supported_ops` 를 풀어 float fallback 을 허용한 설정
   - onnx2tf `-oiqt` (flatbuffer_direct 백엔드) — 애초에 로드 자체가 거부됨
   - onnx2tf `-oiqt -tb tf_converter` — 로드는 되지만 동일하게 무너짐

   → **`--quant int8_hybrid`(가중치 INT8, 3.2MB)가 기본값이며 정상 동작한다.**
     CLAUDE.md §7.4 가 규정한 Hybrid Quantization 경로다.
     NNAPI 가속이 필요한 full-integer 는 Linux 컨테이너(`docker/Dockerfile.export`)에서
     Ultralytics 공식 경로로 재시도 중이다.

5. **경로에 한글이 있으면 안 된다.**
   `ai_edge_litert` / TF 의 C++ 레이어가 비ASCII 경로 파일을 못 연다.
   이 프로젝트 경로(`…/졸푸바오/ai`)가 여기 해당해서, 변환 스크립트들은
   ASCII 임시 디렉터리에서 작업하고 결과만 회수한다.

## 알려진 미해결 사항

- **escalator / automatic_door / kiosk**: 매칭되는 공개 데이터셋 없음 → 직접 촬영 + 수동 라벨링 필요
- **counter**: `reception_desk` 매핑은 찾았으나 현재 박스 1개뿐 → 다른 지역 데이터 추가 확보 필요
- **table / person**: AI Hub 186 에 없음 → `scripts/fetch_coco_subset.py` 로 보강 (미실행)
- 서부산실내 지역: 라벨만 확보, 원천 이미지(11GB) 미다운로드

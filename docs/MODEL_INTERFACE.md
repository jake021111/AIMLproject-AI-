# 모델 연동 스펙 (Flutter ↔ TFLite)

> 대상: Flutter 앱에서 `tflite_flutter` 로 온디바이스 추론을 붙이는 개발자
> 이 문서는 **모델 파일이 바뀌어도 변하지 않는 입·출력 계약**이다.
> 클래스 순서의 단일 소스는 [`configs/class_labels.yaml`](../configs/class_labels.yaml) → `configs/labels.txt`.

---

## 0. 지금 받을 수 있는 것 / 못 받는 것

| 항목 | 상태 |
|---|---|
| 클래스 순서 · 라벨 파일 | ✅ `configs/labels.txt` (§1) |
| 입력 전처리 스펙 | ✅ §2 |
| 출력 파싱 스펙 | ✅ §3 |
| **학습된 실제 `.tflite`** | ✅ **v0.1 제공** — `best_int8_hybrid.tflite` (3.2MB, INT8) |
| 테스트 이미지 + 정답 | ✅ `fixtures/` (§8) |

**v0.1 실측 지표** (test split 156장 / 439박스)

| 지표 | 결과 | 목표 |
|---|---|---|
| mAP@0.5 | **0.826** | ≥ 0.70 |
| veryHigh 평균 recall | **0.910** | ≥ 0.85 |

탐지되는 클래스는 **10개**다 (§7 참조). 나머지 5개는 학습 데이터가 없어 탐지되지 않지만,
**출력 채널은 15클래스(19채널) 그대로**이므로 앱 코드는 처음부터 15클래스 기준으로 짜면 된다.

---

## 1. 클래스 (15개, 순서 고정)

출력 텐서의 클래스 채널 순서 = 아래 id 순서다. **임의로 바꾸면 안 된다.**

| id | label | 한국어 | 우선순위 | conf 임계값 | 거리 추정 |
|---|---|---|---|---|---|
| 0 | chair | 의자 | high | 0.35 | O |
| 1 | table | 테이블 | high | 0.35 | O |
| 2 | pillar | 기둥 | high | 0.35 | O |
| 3 | trash_can | 쓰레기통 | high | 0.35 | O |
| 4 | signage_stand | 안내 표지판 | high | 0.35 | O |
| 5 | person | 사람 | high | 0.35 | O |
| 6 | stairs | 계단 | **veryHigh** | **0.25** | **X** |
| 7 | escalator | 에스컬레이터 | **veryHigh** | **0.25** | O |
| 8 | elevator_door | 엘리베이터 | **veryHigh** | **0.25** | O |
| 9 | door | 문 | high | 0.35 | O |
| 10 | automatic_door | 자동문 | high | 0.35 | O |
| 11 | kiosk | 키오스크 | medium | 0.35 | O |
| 12 | counter | 카운터 | medium | 0.35 | O |
| 13 | vending_machine | 자판기 | medium | 0.35 | O |
| 14 | tactile_paving | 점자블록 | **veryHigh** | **0.25** | **X** |

- `conf 임계값`이 클래스마다 다르다. veryHigh 4종(6, 7, 8, 14)은 미탐(FN)이 치명적이라 0.25로 낮춘다.
- `거리 추정 X`(stairs, tactile_paving)는 bbox 폭과 실제 폭의 대응이 성립하지 않는다.
  거리 수치를 말하지 말고 "전방에 계단이 있습니다" 형태로만 안내한다.

---

## 2. 입력 (전처리)

| 항목 | 값 |
|---|---|
| shape | `[1, 640, 640, 3]` (NHWC) |
| 채널 순서 | **RGB** (BGR 아님) |
| dtype | **현재 배포본: `float32`** (아래 설명 참조) |
| 값 범위 | float 경로는 `0.0 ~ 1.0` (255로 나눔). INT8 경로는 §2.2 |
| 리사이즈 | **letterbox** (비율 유지 + 패딩). 단순 stretch 금지 — bbox가 찌그러진다 |
| 패딩 색 | `(114, 114, 114)` 회색 |

> 현재 배포본은 **INT8 (가중치 INT8 / 활성 float, 이른바 hybrid·dynamic-range 양자화)** 이다.
> 모델 파일은 3.2MB로 작지만 **입출력 텐서는 float32**라서
> **앱에서 양자화/역양자화 코드를 쓸 필요가 없다** (§2.2, §3.2를 건너뛰어도 된다).
>
> 나중에 full-integer INT8(입출력까지 int8)로 교체될 수 있으니,
> `inputTensor.type`을 런타임에 확인해 분기해 두면 그때도 코드를 안 고쳐도 된다.

### 2.1 letterbox

```dart
// 원본 w x h -> 640 x 640
final scale = math.min(640 / srcW, 640 / srcH);
final newW  = (srcW * scale).round();
final newH  = (srcH * scale).round();
final padX  = (640 - newW) / 2.0;   // 좌우 균등 패딩
final padY  = (640 - newH) / 2.0;

// scale, padX, padY 는 출력 좌표 역변환(§3.4)에 반드시 필요하다. 프레임마다 보관할 것.
```

720p(1280x720) 입력이면 `scale = 0.5`, `padX = 0`, `padY = 140` 으로 고정이다.
카메라 해상도를 바꾸지 않는 한 매 프레임 재계산할 필요는 없다 (할당 줄이기).

### 2.2 INT8 양자화 입력 (중요)

INT8 모델의 입력 텐서는 양자화되어 있다. **파라미터를 하드코딩하지 말고 런타임에 읽는다.**

```dart
final inputTensor = interpreter.getInputTensor(0);
final inScale = inputTensor.params.scale;      // 보통 1/255 = 0.003921...
final inZero  = inputTensor.params.zeroPoint;  // int8이면 보통 -128, uint8이면 0

// pixel(0~255) -> 양자화 값
final q = (pixel / 255.0 / inScale).round() + inZero;
// inScale == 1/255 이고 inZero == -128 인 일반적인 경우, 결과적으로:  q = pixel - 128
```

`inputTensor.type` 으로 `int8` / `uint8` / `float32` 를 분기해두면 FP16 fallback 모델(§5)도 같은 코드로 돌아간다.

---

## 3. 출력 (파싱)

### 3.1 shape

```
[1, 19, 8400]      19 = 4(bbox) + 15(classes),  8400 = 앵커 후보 수
```

**전치(transpose)되어 있다.** 흔한 착각처럼 `[1, 8400, 19]` 가 아니다.
`output[0][row][anchor]` 순서로 인덱싱된다.

| row | 의미 |
|---|---|
| 0 | `cx` — 박스 중심 x, **0~1 정규화 (letterbox 640 좌표계 기준)** |
| 1 | `cy` — 박스 중심 y, 0~1 정규화 |
| 2 | `w` — 박스 폭, 0~1 정규화 |
| 3 | `h` — 박스 높이, 0~1 정규화 |
| 4~18 | 클래스 0~14 의 confidence |

- YOLOv8은 **objectness 채널이 없다.** row 4~18이 곧 최종 클래스 점수이며 이미 sigmoid가 적용된 0~1 값이다.
  (v5처럼 `obj * cls` 를 곱하면 안 된다.)
- 정규화 기준이 **원본 이미지가 아니라 letterbox된 640 좌표계**라는 점이 핵심이다. §3.4에서 역변환한다.

### 3.2 dequantize

```dart
final outTensor = interpreter.getOutputTensor(0);
final outScale = outTensor.params.scale;
final outZero  = outTensor.params.zeroPoint;

double deq(int raw) => (raw - outZero) * outScale;
```

### 3.3 후보 추출

```dart
const anchors = 8400;
final results = <Candidate>[];

for (var i = 0; i < anchors; i++) {
  // 클래스별 max 찾기
  var bestId = -1;
  var bestScore = 0.0;
  for (var c = 0; c < numClasses; c++) {
    final s = deq(raw[4 + c][i]);
    if (s > bestScore) { bestScore = s; bestId = c; }
  }
  if (bestId < 0) continue;

  // 클래스별 임계값 (veryHigh 4종은 0.25, 나머지 0.35)
  if (bestScore < classInfo[bestId].confThreshold) continue;

  results.add(Candidate(bestId, bestScore,
      deq(raw[0][i]), deq(raw[1][i]), deq(raw[2][i]), deq(raw[3][i])));
}
```

> 성능 메모: 8400 x 15 = 126,000회 루프다. 프레임당 객체를 새로 할당하지 말고
> 버퍼를 재사용하고, 이 파싱까지 isolate 안에서 끝낸다 (CLAUDE.md §5.2).

### 3.4 letterbox 역변환 → 원본 픽셀 좌표

```dart
// 1) 정규화 -> letterbox 640 픽셀
final cxLb = cx * 640, cyLb = cy * 640;
final wLb  = w  * 640, hLb  = h  * 640;

// 2) 패딩 제거 + 스케일 되돌리기 -> 원본 이미지 픽셀
final cxSrc = (cxLb - padX) / scale;
final cySrc = (cyLb - padY) / scale;
final wSrc  = wLb / scale;
final hSrc  = hLb / scale;

// 3) cx,cy,w,h -> Rect(LTRB), 이미지 경계로 clamp
final rect = Rect.fromLTRB(
  (cxSrc - wSrc / 2).clamp(0.0, srcW.toDouble()),
  (cySrc - hSrc / 2).clamp(0.0, srcH.toDouble()),
  (cxSrc + wSrc / 2).clamp(0.0, srcW.toDouble()),
  (cySrc + hSrc / 2).clamp(0.0, srcH.toDouble()),
);
```

### 3.5 NMS

- 모델에 NMS가 **포함되어 있지 않다** (`nms=False` 로 export — INT8 양자화 호환성 문제).
  Dart 쪽에서 직접 돌려야 한다.
- **클래스별로** 수행한다 (서로 다른 클래스의 박스는 겹쳐도 지우지 않는다).
- IoU 임계값 **0.45**.

```
1. 후보를 confidence 내림차순 정렬
2. 최고점 박스 채택 -> 같은 classId 이고 IoU > 0.45 인 나머지 제거
3. 남은 후보로 반복
```

### 3.6 검증 체크리스트

파이프라인이 맞게 붙었는지 확인하는 순서 (스텁 모델로도 1~3은 검증 가능):

1. 입력 텐서 shape `[1,640,640,3]`, 출력 텐서 shape `[1,19,8400]` 이 로그로 찍히는가
2. dequantize 후 row 4~18 값이 `0.0 ~ 1.0` 범위에 들어오는가 (음수나 100 같은 값이면 dequantize 오류)
3. 역변환한 Rect가 원본 이미지 범위 안에 있고, 폭·높이가 양수인가
4. (실제 모델) 정면 촬영한 문·기둥에 박스가 잡히는가, 화면 좌측 객체의 `angleDeg` 가 음수인가

---

## 4. 파일 배치 (Flutter 측)

```
assets/
├── models/yolov8n_indoor_int8.tflite   # 모델 (이름 고정)
└── labels.txt                          # configs/labels.txt 를 그대로 복사
```

`labels.txt` 는 AI 저장소의 `configs/labels.txt` 가 원본이다.
클래스가 바뀌면 AI 쪽에서 재생성해 전달하므로, 앱에서 손으로 편집하지 않는다.

라인 수(15)와 모델 출력 채널 수(19 - 4 = 15)가 일치하는지 로드 시 assert 하면 사고를 막을 수 있다.

---

## 5. 델리게이트 / 성능

| 모델 | 크기 | 입출력 | 가속 | 현재 |
|---|---|---|---|---|
| **INT8 hybrid** (가중치 INT8) | **3.2 MB** | float32 | CPU / XNNPACK | ✅ **현재 배포본** |
| INT8 full integer | 3.4 MB | int8 | NNAPI (NPU/DSP) | 🔄 변환 작업 중 |

full-integer INT8이 최종 목표(NNAPI 가속)이고 변환 작업이 진행 중이다.
교체되더라도 **shape · 좌표계 · 클래스 순서 · NMS는 전혀 바뀌지 않는다.**
달라지는 건 텐서 dtype(`float32` → `int8`)뿐이고, §2.2 / §3.2의 양자화 코드가 그때 필요해진다.

> 그래서 **클래스 수를 `outputShape[1] - 4`로 유도하고 dtype을 런타임에 분기**해 두라고 권한다.
> 그렇게 짜두면 교체 시 코드 수정이 0이다.

- 추론은 `IsolateInterpreter` 로 UI 스레드와 분리한다.
- 카메라 30fps를 전부 추론하지 않는다. `_isProcessing` 플래그로 프레임 드롭.
- 목표: 최소 10 FPS, 탐지에서 음성 출력까지 1초 이내.

---

## 6. 연동 검증용 스텁 모델

학습이 끝나기 전에 프론트 연동을 진행하기 위한 **구조만 동일한 모델**이다.

```bash
# Linux x86 / macOS
python scripts/export_stub_model.py --mode coco80   # 디코드 정확성 검증용 (실가중치)
python scripts/export_stub_model.py --mode stub15   # 15-class 배선 검증용 (랜덤 가중치)

# Windows — Ultralytics 가 TFLite export 를 막아 두므로 ONNX 경유로 변환한다
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, nms=False, opset=13)"
python scripts/onnx_to_tflite.py --onnx yolov8n.onnx
```

- (A)는 출력이 `[1, 19, 8400]` 으로 최종 모델과 **완전히 동일**하다. 파일만 교체하면 끝난다.
- (B)는 출력이 `[1, 84, 8400]` (80 classes)이다. 클래스 수를 상수로 박아두지 말고
  **출력 텐서 shape에서 `numClasses = shape[1] - 4` 로 유도**하면 (A)/(B)/최종 모델을 코드 변경 없이 전환할 수 있다.
  COCO의 `person`(id 0), `chair`(id 56)로 박스 위치가 정확한지 눈으로 확인하는 용도다.

권장 순서: **(B)로 디코드 정확성을 잡고 → (A)로 15-class 배선을 맞춘 뒤 → 최종 모델 교체.**

---

## 7. 모델 전달 일정

| 단계 | 내용 | 클래스 | 상태 |
|---|---|---|---|
| 스텁 (COCO 80) | 디코드 로직 검증용 | 80 (COCO) | ✅ 완료 |
| **v0.1** | AI Hub 실내 데이터로 학습한 첫 실모델 | **10** | ✅ **전달** (mAP@0.5 0.826) |
| v1.0 | 15 클래스 전체 | 15 | 데이터 확보 후 |

**v0.1 클래스별 성능 (test split)**

| 클래스 | recall | mAP@0.5 | 비고 |
|---|---|---|---|
| trash_can | 0.972 | 0.989 | |
| elevator_door | 0.919 | 0.977 | veryHigh |
| stairs | 1.000 | 0.978 | veryHigh |
| vending_machine | 0.935 | 0.973 | |
| pillar | 0.885 | 0.906 | |
| door | 0.857 | 0.789 | |
| tactile_paving | 0.812 | 0.843 | veryHigh |
| signage_stand | 0.499 | 0.562 | 학습 인스턴스 55개, 약함 |
| chair | 0.429 | 0.414 | 학습 인스턴스 71개, 약함 |
| table, person, escalator, automatic_door, kiosk, counter | — | — | **학습 데이터 없음 → 탐지 안 됨** |

**v0.1에서 5개 클래스가 빠지는 이유:** AI Hub 186 데이터셋에 대응 카테고리가 없다.

- `table`, `person` → COCO 2017에서 추출 (`scripts/fetch_coco_subset.py` 작성됨, 미실행)
- `escalator`, `automatic_door`, `kiosk` → 공개 데이터셋 없음. 직접 촬영 + 라벨링 필요
- `counter` → 매핑은 찾았으나 현재 박스 1개뿐이라 데이터 보강 필요

v0.1도 **출력 채널은 19로 유지**한다 (15-class 헤드 그대로 학습하고, 데이터가 없는 클래스는 탐지가 안 될 뿐이다).
따라서 앱 코드는 처음부터 15-class 기준으로 작성하면 되고, v0.1에서 v1.0으로 교체할 때도 코드 변경이 없다.

---

## 8. 테스트 픽스처

`scripts/make_test_fixtures.py` 로 생성한다.

```
fixtures/
├── images/*.jpg                  # AI Hub 실내 촬영 이미지
└── expected_detections.json      # 각 이미지의 정답(ground truth) 박스
```

`expected_detections.json` 은 **사람이 라벨링한 정답값**이며 모델 출력이 아니다.
모델이 이것과 완전히 일치할 수는 없고, 다음 확인용으로 쓴다.

- 좌표 규약이 맞는지 (같은 객체를 비슷한 위치에 잡는지)
- 클래스 id 배선이 맞는지 (문을 `door`로 부르는지, 엉뚱한 라벨이 나오는지)

이미지는 AI Hub 데이터라 **공개 저장소(GitHub)에 올리지 않는다.** `fixtures/` 는 gitignore 대상이며
팀 드라이브 등으로 따로 전달한다.

### expected_detections.json 포맷

```json
{
  "note": "ground truth (사람 라벨). 모델 출력 아님.",
  "images": [
    {
      "file": "images/example.jpg",
      "width": 1920,
      "height": 1080,
      "objects": [
        {
          "class_id": 9,
          "label": "door",
          "bbox_norm": { "cx": 0.35, "cy": 0.54, "w": 0.04, "h": 0.42 },
          "bbox_pixel": { "left": 633.6, "top": 356.4, "right": 710.4, "bottom": 810.0 }
        }
      ]
    }
  ]
}
```

`bbox_norm` 은 **원본 이미지 기준** 0~1 정규화값이다 (letterbox 좌표계가 아니다).
§3.4 역변환을 거친 앱의 최종 Rect와 `bbox_pixel` 을 비교하면 된다.

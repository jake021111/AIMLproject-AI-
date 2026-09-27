# AGENTS.md — insight (시각장애인 실내 보행 지원 AI 앱)

> 이 파일은 Codex가 이 프로젝트에서 작업할 때 항상 참조하는 컨텍스트 문서다.
> 프로젝트 루트에 두고, 새 세션마다 자동으로 읽힌다.

---

## 0. 프로젝트 한 줄 요약

시각장애인이 안내견이나 타인의 도움 없이 실내 시설을 독립적으로 이용할 수 있도록,
**온디바이스 객체 탐지 + 거리/방향 안내 + 음성 인터페이스**를 제공하는 Flutter 모바일 앱.

- 팀명: insight
- 과목: AIML응용프로젝트 (졸업프로젝트)
- 핵심 가치: **실시간성**(끊기면 위험), **간결한 안내**(장황하면 방해), **저전력**(오래 써야 함)

---

## 1. 기술 스택 (확정)

| 영역 | 선택 | 비고 |
|---|---|---|
| 앱 프레임워크 | Flutter 3.x (Dart 3.x) | Android 우선, iOS는 2차 |
| 상태관리 | **Riverpod** (flutter_riverpod) | 서비스 계층 DI까지 함께 처리 |
| 카메라 | `camera` | `ImageFormatGroup.yuv420` (Android) / `bgra8888` (iOS) |
| 객체 탐지 | **YOLOv8n → TFLite INT8** + `tflite_flutter` | `IsolateInterpreter`로 UI 스레드 분리 |
| 거리 추정 | ① ARCore Depth (Platform Channel) ② Bounding Box 비례식 | 상세는 §6 |
| OCR | `google_mlkit_text_recognition` (온디바이스, 한국어) | 무료·오프라인·저지연 |
| STT | `speech_to_text` (OS 엔진) | |
| TTS | `flutter_tts` (OS 엔진) | 속도/음색 설정 지원 |
| 햅틱 | `vibration` | 위험 경고용 강한 연속 진동 |
| 로컬 저장 | `shared_preferences` | 사용자 설정 |
| 서버 통신 | `dio` | 경고 로그 / 설정 동기화 |
| 백엔드 | Spring Boot + MySQL | §9 참조 |
| 학습 환경 | Python 3.10+, `ultralytics`, PyTorch | §7 참조 |

### 패키지 선택 원칙
- **온디바이스 우선.** 클라우드 API는 네트워크 끊기면 죽는다. 시각장애인 보행 중 네트워크 끊김 = 서비스 중단이므로 OCR/STT/TTS 모두 OS·온디바이스 엔진 사용.
- 상시 구동되는 신경망은 **YOLO 하나뿐**. 두 번째 딥러닝 모델을 동시에 돌리지 않는다.

---

## 2. 폴더 구조

```
lib/
├── main.dart
├── app.dart                        # MaterialApp, 라우팅, 테마
├── core/
│   ├── constants/
│   │   ├── class_labels.dart       # 15개 클래스 정의 + 우선순위 + 실측 크기
│   │   ├── thresholds.dart         # confidence, NMS IoU, 경고 거리 기본값
│   │   └── speech_templates.dart   # TTS 문장 템플릿
│   ├── utils/
│   │   ├── image_converter.dart    # YUV420 → RGB (isolate)
│   │   ├── nms.dart                # Non-Maximum Suppression
│   │   └── geometry.dart           # 각도/시계방향 변환
│   └── result.dart                 # Result<T, E> 래퍼
├── data/
│   ├── models/
│   │   ├── detected_object.dart
│   │   ├── user_settings.dart
│   │   └── warning_log.dart
│   ├── datasources/
│   │   ├── local/settings_local_ds.dart
│   │   └── remote/api_client.dart
│   └── repositories/
│       ├── settings_repository.dart
│       └── warning_log_repository.dart
├── services/
│   ├── camera_service.dart         # 프레임 스트림 + 스로틀링
│   ├── detection_service.dart      # TFLite 추론 + 후처리
│   ├── distance_service.dart       # 거리 추정 (전략 패턴)
│   ├── direction_service.dart      # 방향(시계) 산출
│   ├── warning_service.dart        # 위험 판단 + 인터럽트
│   ├── tts_service.dart            # 우선순위 큐 기반
│   ├── stt_service.dart
│   ├── ocr_service.dart
│   └── haptic_service.dart
├── state/
│   ├── detection_provider.dart
│   ├── settings_provider.dart
│   └── voice_provider.dart
└── presentation/
    ├── screens/
    │   ├── onboarding/
    │   ├── camera_navigation/      # 메인
    │   ├── voice_assistant/
    │   ├── ocr_reader/
    │   └── settings/
    └── widgets/
        ├── detection_overlay.dart  # bbox + 라벨 렌더링
        └── accessible_button.dart  # 큰 터치 영역 + Semantics

assets/
├── models/
│   └── yolov8n_indoor_int8.tflite
└── labels.txt

android/app/src/main/kotlin/.../
└── DepthChannel.kt                 # ARCore Depth API 브릿지 (Phase 3)

ai/                                 # 학습·변환 (앱과 분리)
├── data.yaml
├── train.py
├── export.py
└── validate.py
```

---

## 3. 도메인 모델

### 3.1 클래스 정의 (15개)

```dart
// lib/core/constants/class_labels.dart
enum GuidePriority { veryHigh, high, medium }

class ClassInfo {
  final int id;
  final String label;        // 모델 라벨 (영문)
  final String korean;       // TTS 출력용
  final GuidePriority priority;
  final double realWidthM;   // bbox 거리 추정용 평균 실측 폭 (m)
  final bool distanceReliable; // false면 거리 대신 "전방에 ○○" 로만 안내
}
```

| id | label | 한국어 | 카테고리 | 우선순위 | 실측 폭(m) | 거리추정 |
|---|---|---|---|---|---|---|
| 0 | chair | 의자 | 충돌(정적) | high | 0.50 | O |
| 1 | table | 테이블 | 충돌(정적) | high | 1.20 | O |
| 2 | pillar | 기둥 | 충돌(정적) | high | 0.50 | O |
| 3 | trash_can | 쓰레기통 | 충돌(정적) | high | 0.40 | O |
| 4 | signage_stand | 안내 표지판 | 충돌(정적) | high | 0.60 | O |
| 5 | person | 사람 | 충돌(동적) | high | 0.50 | O |
| 6 | stairs | 계단 | 수직이동 | **veryHigh** | — | X |
| 7 | escalator | 에스컬레이터 | 수직이동 | **veryHigh** | 1.00 | O |
| 8 | elevator_door | 엘리베이터 | 수직이동 | **veryHigh** | 1.00 | O |
| 9 | door | 문 | 출입구 | high | 0.90 | O |
| 10 | automatic_door | 자동문 | 출입구 | high | 1.20 | O |
| 11 | kiosk | 키오스크 | 시설물 | medium | 0.60 | O |
| 12 | counter | 카운터 | 시설물 | medium | 1.50 | O |
| 13 | vending_machine | 자판기 | 시설물 | medium | 1.00 | O |
| 14 | tactile_paving | 점자블록 | 보행보조 | **veryHigh** | — | X |

> `stairs`, `tactile_paving`은 바닥에 깔린 넓은 영역이라 bbox 폭↔실제 폭 대응이 성립하지 않는다.
> 거리 수치를 말하지 말고 **"전방에 계단이 있습니다"** 처럼만 안내한다.

### 3.2 DetectedObject

```dart
class DetectedObject {
  final int classId;
  final String label;
  final double confidence;
  final Rect bbox;              // 원본 이미지 좌표계 (픽셀)
  final double? distanceM;      // null = 추정 실패
  final double? nearestDistanceM; // P10 (충돌 판단용)
  final DistanceSource source;  // arcore | bboxRatio | none
  final bool lowConfidenceDistance; // 유효 depth 픽셀 부족 시 true
  final double angleDeg;        // -FoV/2 ~ +FoV/2 (좌:음수, 우:양수)
  final int clockDirection;     // 9~12~3
}

enum DistanceSource { arcore, bboxRatio, none }
```

### 3.3 UserSettings

```dart
class UserSettings {
  final double warningDistanceM;  // 기본 2.0, 범위 0.5~5.0
  final double ttsSpeed;          // 기본 0.5, 범위 0.1~1.0
  final String voiceType;         // 'female' | 'male'
  final bool hapticEnabled;       // 기본 true
  final String language;          // 'ko-KR'
  final String deviceUuid;
}
```

---

## 4. 시스템 아키텍처 (3계층)

```
Presentation Layer   화면 + Overlay + 접근성 위젯
        ↓ (Riverpod Provider)
Business Logic Layer DetectionController / WarningController / SettingsController
        ↓
Service Layer        Camera / Detection / Distance / TTS / STT / OCR / Haptic
        ↓
External Layer       TFLite 모델 · ARCore Depth · OS STT/TTS · ML Kit OCR · Backend
```

**의존 규칙:** 위 → 아래 단방향만 허용. Service는 Presentation을 모른다.
Service는 인터페이스(abstract class)로 정의하고 구현체를 주입한다 (테스트 대체 용이).

---

## 5. 실시간 탐지 파이프라인

### 5.1 전체 흐름

```
CameraImage(YUV420, 1280x720, 30fps)
   → [스로틀: 6~10 FPS 샘플링]
   → Isolate: YUV420 → RGB → 640x640 letterbox → Float/UInt8 입력
   → IsolateInterpreter.run()  (YOLOv8n INT8)
   → 출력 dequantize → [8400, 4+15] 디코드
   → confidence filter (≥ 0.35) → NMS (IoU 0.45)
   → DistanceService.estimate()  → DetectedObject 리스트
   → DirectionService.compute()
   → WarningService.evaluate()   → 위험 시 인터럽트
   → UI Overlay 렌더링 + TTS 큐 투입
```

### 5.2 성능 규칙 (반드시 지킬 것)

- **추론은 절대 UI 스레드에서 하지 않는다.** `IsolateInterpreter` 사용.
- 카메라 30fps를 전부 추론하지 않는다. `_isProcessing` 플래그로 **프레임 드롭** 처리:
  ```dart
  if (_isProcessing) return;   // 이전 프레임 처리 중이면 그냥 버림
  _isProcessing = true;
  ...
  finally { _isProcessing = false; }
  ```
- YUV420 → RGB 변환은 Dart 루프로 하면 프레임당 수십 ms 잡아먹는다.
  → isolate에서 처리하고, 가능하면 카메라 해상도를 `ResolutionPreset.medium`(720p)로 낮춘다.
- 목표: **최소 10 FPS, 목표 15~20 FPS**, 탐지→음성 출력까지 **1초 이내**.
- 발열 관리: 가속도계로 **사용자가 정지 상태면 추론 주기를 2~3 FPS로 낮춘다** (선택 구현).

### 5.3 YOLOv8 출력 파싱 (중요)

YOLOv8 TFLite 출력 shape은 보통 `[1, 19, 8400]` (= 4 bbox + 15 classes, **transpose 되어 있음**).

```dart
// 1) INT8 dequantize
final double scale  = outputTensor.params.scale;
final int zeroPoint = outputTensor.params.zeroPoint;
double v = (raw[i] - zeroPoint) * scale;

// 2) [19, 8400] → 8400개 후보로 전치
// 행 0~3: cx, cy, w, h (0~1 정규화)
// 행 4~18: 클래스별 confidence (objectness 없음, v8은 sigmoid 된 cls score 직접)

// 3) maxScore >= confThreshold 인 것만 취하고
// 4) letterbox 역변환으로 원본 좌표 복원
// 5) 클래스별 NMS (IoU 0.45)
```

> NMS는 모델 밖(Dart)에서 수행한다. export 시 NMS를 모델에 포함시키지 않는다 (INT8 양자화 호환성 문제).

---

## 6. 거리 추정 (설계 핵심)

### 6.1 전략: 2단계 하이브리드

| 순위 | 방식 | 조건 | 결과 |
|---|---|---|---|
| 1 | **ARCore Depth API** | ARCore Depth 지원 기기 | 절대 거리(m), 정확 |
| 2 | **Bounding Box 비례식** | 그 외 모든 기기 | 근사 거리, "약 ○미터" 로 안내 |

> ❗ **Depth Anything v2 같은 monocular depth 모델은 온디바이스 상시 구동하지 않는다.**
> YOLO와 동시에 두 개의 신경망을 돌리면 모바일에서 10 FPS도 못 나오고 발열이 심하다.
> 향후 정확도 고도화 단계에서 선택적으로만 검토한다.

**구현 순서:** Phase 2에서 ②(순수 Dart, 네이티브 코드 불필요)부터 만들고, Phase 3에서 ①을 Platform Channel로 추가한다.

### 6.2 Bounding Box 비례식

```dart
// 초점거리(픽셀) = (이미지폭/2) / tan(수평화각/2)
final focalPx = (imageWidth / 2) / math.tan(_toRad(cameraFovDeg) / 2);

// 거리 = (실제 폭 × 초점거리) / bbox 픽셀 폭
final distanceM = (classInfo.realWidthM * focalPx) / bbox.width;
```

- `cameraFovDeg`는 기기마다 다르다. `camera` 플러그인에서 직접 못 얻으므로 **기본값 67°** 로 두고 상수화한다.
- 이 방식은 객체가 정면을 보고 있다고 가정한다 → 오차 ±30% 수준. **반드시 "약"을 붙여 안내.**
- `distanceReliable == false` 클래스(계단, 점자블록)는 이 계산을 건너뛴다.

### 6.3 ARCore Depth: Bounding Box × Depth Map 매핑

```
1. bbox 영역에서 중앙 50% (inner ROI)만 추출  → 모서리 배경 픽셀 제거
2. inner ROI 내 모든 유효 depth 픽셀의 중앙값(Median) = 대표 거리
3. 같은 ROI의 하위 10% 분위수(P10) = "가장 가까운 지점" → 충돌 판단에 사용
4. 유효 depth 픽셀이 ROI의 30% 미만이면 lowConfidenceDistance = true
   → 거리 수치 대신 "전방에 ○○ 있음" 으로 표현
```

**왜 Median인가**

| 방식 | 장점 | 단점 | 채택 |
|---|---|---|---|
| 중심 픽셀 | 계산 빠름 | 중심이 빈 공간(의자 등받이 사이)이면 배경 거리가 잡힘 | ✗ |
| 평균(Mean) | 전체 영역 반영 | 배경 픽셀/이상치에 민감 | ✗ |
| **중앙값(Median)** | **이상치에 강건, 객체 본체의 대표 거리** | 정렬 비용 약간 | **✓** |
| 최솟값(Min) | "가장 가까운 부분" | 노이즈 픽셀 하나에 좌우됨 | △ (P10으로 대체) |

### 6.4 방향 산출

```dart
// 선형 근사식 (안내 용도로는 충분)
final angleDeg = (bbox.center.dx / imageWidth - 0.5) * cameraFovDeg;

// 시계 방향 매핑: 12시=정면, 9시=좌측, 3시=우측
// FoV 67° 기준 대략 10시~2시 범위만 표현 가능
int toClock(double angleDeg) {
  if (angleDeg < -20) return 10;
  if (angleDeg < -7)  return 11;
  if (angleDeg <= 7)  return 12;
  if (angleDeg <= 20) return 1;
  return 2;
}
```

---

## 7. AI 모델 학습 · 변환 (`ai/` 디렉터리)

### 7.1 데이터셋

- **AI Hub 공개 데이터셋**: 실내 공간 객체 인식 이미지, 시각장애인 보행 환경 데이터 (한국어 시설명·실내 환경 포함)
- **COCO 2017 일부**: chair, person, dining table 등 겹치는 클래스만 선별
- ⚠️ 두 출처의 **라벨 taxonomy가 다르므로 통합 매핑 테이블을 먼저 만든다** (`ai/label_map.yaml`)
- 분할: train 70% / val 20% / test 10%

### 7.2 Recall 우선 전략

우선순위 `veryHigh` 클래스(stairs, tactile_paving, escalator, elevator_door)는
**오탐(FP)보다 미탐(FN)이 치명적**이다.

```
❌ 하지 말 것: YOLOv8 학습에 per-class weight 인자를 넘기려 시도
   (Ultralytics 기본 학습 파이프라인에는 그런 옵션이 없다. cls loss가 BCE라
    직접 넣으려면 loss 코드를 수정해야 한다.)

✅ 할 것:
   1. veryHigh 클래스가 포함된 이미지를 데이터셋 구성 단계에서 2배 오버샘플링
   2. copy-paste / mosaic 증강으로 해당 클래스 인스턴스 수 보강
   3. 추론 시 veryHigh 클래스만 confidence threshold를 0.25로 낮춤 (나머지 0.35)
```

### 7.3 학습

```python
# ai/train.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # COCO 2017 80-class 사전학습 가중치
model.train(
    data="data.yaml",
    imgsz=640,
    batch=32,
    epochs=150,
    patience=20,            # Early Stopping
    optimizer="SGD",
    momentum=0.937,
    lr0=0.01,
    lrf=0.01,               # ⚠️ lrf는 '비율' → 최종 LR = lr0 × lrf = 0.0001
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    close_mosaic=10,        # 마지막 10 epoch mosaic 비활성화 (수렴 안정)
    freeze=10,              # ⚠️ '앞 10개 레이어'를 고정 (= backbone 저수준 feature)
                            #    '10 epoch 동안'이 아니다. 학습 전체에 걸쳐 고정된다.
    project="runs", name="indoor15",
)
```

> `freeze`는 **레이어 개수**다. Ultralytics는 학습 도중 자동 unfreeze를 하지 않는다.
> 정말 2단계(freeze → unfreeze)로 하고 싶으면 train()을 두 번 호출해야 한다. MVP에서는 1회로 충분.

**목표 지표:** mAP@0.5 ≥ 0.70 (실내 테스트 기준), veryHigh 클래스 recall ≥ 0.85

### 7.4 TFLite INT8 변환

```bash
# 수동 4단계(ONNX→TF→TFLite) 파이프라인을 직접 짤 필요 없다.
# 아래 한 줄이 내부적으로 PyTorch → ONNX → TF SavedModel → TFLite INT8 + PTQ를 전부 수행한다.
yolo export model=runs/indoor15/weights/best.pt format=tflite int8=True imgsz=640 nms=False
```

- 대표 데이터셋(Representative Dataset): 학습 데이터에서 무작위 300장 → `data.yaml`의 val 경로에서 자동 샘플링
- Full Integer Quantization (weight + activation 모두 INT8) → **NPU/DSP delegate(NNAPI)** 호환
- ⚠️ INT8은 **GPU delegate에서 가속되지 않는다.** GPU를 쓰려면 FP16 경로를 따로 뽑아야 한다.
- 정확도 손실이 5%를 초과하면 → Hybrid Quantization(weight INT8 / activation FP16)으로 fallback

**크기 기준선**

| 형식 | 크기 | 비고 |
|---|---|---|
| best.pt (PyTorch) | ~6 MB | |
| ONNX FP32 | ~12 MB | |
| TFLite FP16 | ~6 MB | |
| **TFLite INT8** | **~3 MB** | 채택 |

### 7.5 검증

```bash
yolo val model=best.pt data=data.yaml split=test
# INT8 변환 후에도 동일하게 val 돌려서 mAP 낙폭(-3~5% 허용) 확인
yolo val model=best_saved_model/best_int8.tflite data=data.yaml split=test
```

---

## 8. 음성 인터페이스 규칙

### 8.1 TTS 우선순위 큐

```dart
enum SpeechPriority {
  critical,  // 충돌 경고 — 진행 중인 발화를 즉시 중단(interrupt)하고 끼어듦
  normal,    // 일반 탐지 안내, 챗봇 응답
  low,       // 설정 변경 확인음 등
}
```

- `critical`이 들어오면 `flutter_tts.stop()` 후 즉시 재생. 기존 큐는 비운다.
- `normal`은 큐에 쌓되 **최대 2개까지만** 유지 (오래된 것부터 버림). 안 그러면 안내가 밀려서 현실과 어긋난다.

### 8.2 안내 문장 템플릿 (짧게!)

```dart
// ✅ 좋은 예
"11시 방향 2미터, 의자"
"전방에 계단이 있습니다"
"정면 1미터, 사람. 주의하세요"

// ❌ 나쁜 예 (비기능 요구사항 위반 — 불필요한 시각적 묘사 금지)
"앞쪽에 갈색 나무 재질의 등받이가 있는 의자가 놓여 있고 그 뒤로 창문이 보입니다"
```

규칙:
- 순서는 **방향 → 거리 → 객체**. 위험할 때만 뒤에 "주의하세요" 추가.
- 색상·재질·배경 묘사 금지.
- 거리 신뢰도가 낮으면(`lowConfidenceDistance`) 수치를 빼고 "전방에 ○○ 있음".
- bbox 비례식으로 추정한 거리는 **"약 2미터"** 처럼 근사임을 명시.

### 8.3 중복 안내 방지

같은 클래스를 매 프레임 말하면 못 쓴다.

```dart
// 클래스별 쿨다운: 같은 클래스는 3초 내 재안내 금지
// 단, 거리가 이전 안내 대비 30% 이상 가까워졌으면 쿨다운 무시
```

---

## 9. 백엔드 API (Spring Boot)

### 9.1 위험 경고 로그 저장

```
POST /warnings/log
Request  { user_id: int, detected_object: string, object_distance: float }
Response { status: string, message: string, data: { log_id: int } }
```

### 9.2 사용자 환경설정 업데이트

```
PATCH /users/{device_uuid}/settings
Request  { warning_distance: float, tts_speed: float, voice_type: string }
Response { status: string, message: string }
```

### 9.3 DB 스키마 (MVP)

```sql
CREATE TABLE users (
  user_id          BIGSERIAL PRIMARY KEY,
  device_uuid      VARCHAR(64) UNIQUE NOT NULL,
  warning_distance NUMERIC(3,1) DEFAULT 2.0,
  tts_speed        NUMERIC(3,2) DEFAULT 0.5,
  voice_type       VARCHAR(16)  DEFAULT 'female',
  created_at       TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE warning_logs (
  log_id          BIGSERIAL PRIMARY KEY,
  user_id         BIGINT REFERENCES users(user_id),
  detected_object VARCHAR(32)  NOT NULL,
  object_distance NUMERIC(4,2),
  created_at      TIMESTAMP    DEFAULT NOW()
);
```

> ⚠️ **로그 전송이 실패해도 앱은 절대 멈추지 않는다.** fire-and-forget + 로컬 큐잉 후 재시도.
> 보행 안내가 네트워크에 의존하면 안 된다.

---

## 10. 화면별 명세

| 화면 | 상태(State Machine) | 핵심 동작 |
|---|---|---|
| **Onboarding** | — | 서비스 소개 → 카메라/마이크 권한 요청 |
| **Camera Navigation** (메인) | `Idle → Detecting → ObjectDetected → Warning → Safe → Detecting` | 실시간 탐지 + Overlay + 음성 안내. 하단에 음성안내/음성비서/OCR 3버튼 |
| **Voice Assistant** | `Idle → Listening → Processing → Response` | STT 입력 → 의도 분석 → TTS 응답 |
| **OCR Reader** | `CameraReady → Scanning → TextDetected → Reading → Complete` | 키오스크/안내판 촬영 → 텍스트 추출 → 읽어주기 |
| **Settings** | — | 음성 속도, 진동 알림, 경고 거리, 언어 |

### 접근성 (반드시 지킬 것)
- 모든 인터랙티브 위젯에 `Semantics(label: ...)` 지정.
- 터치 타겟 최소 **48×48 dp**, 메인 화면 3버튼은 화면 하단 1/3을 3등분해 크게 배치.
- TalkBack/VoiceOver로 조작 가능해야 한다. **음성만으로 모든 기능 조작 가능**이 필수 요구사항.
- 화면 진입 시 해당 화면의 목적을 자동 TTS 안내.
- 시각적 Overlay(bbox)는 **개발자 디버깅용**이며, 최종 사용자는 못 본다는 걸 전제로 설계한다.

---

## 11. 개발 로드맵 (MVP 우선 구현)

| Phase | 범위 | 완료 기준 |
|---|---|---|
| **1** | Flutter UI/UX 골격 | 5개 화면 + Navigation + Settings 저장/로드 동작 |
| **2** | YOLOv8n + TFLite 연동 | 카메라 프리뷰 위 bbox 표시, 10 FPS 이상, **bbox 비례식 거리 + TTS 안내** |
| **3** | OCR + 음성 인터페이스 | ML Kit OCR 읽어주기, STT 명령 인식, ARCore Depth Platform Channel 추가 |
| **4** | 백엔드 연동 + 고도화 | 경고 로그 전송, 설정 동기화, (선택) LMM 기반 상황 설명 |

**현재 위치:** Phase 1 완료 (UI 프로토타입), **Phase 2 착수**

---

## 12. 코딩 컨벤션

- **Dart**: `flutter_lints` 기본 + `always_declare_return_types`. 포맷은 `dart format` (80자 아님, 100자 허용).
- 네이밍: 파일 `snake_case.dart`, 클래스 `PascalCase`, 상수 `lowerCamelCase`.
- 서비스는 `abstract class XService` + `class XServiceImpl` 쌍으로 만든다.
- 에러는 throw보다 `Result<T, AppError>` 반환 선호. **특히 탐지 루프 안에서는 절대 throw하지 않는다** (프레임 하나 실패로 전체가 죽으면 안 됨).
- `print()` 금지 → `debugPrint()` 또는 `logger`.
- 매직 넘버 금지. 임계값은 전부 `core/constants/thresholds.dart`.

### 테스트
- `test/` 에 단위 테스트: NMS, 거리 계산, 각도→시계 변환, TTS 템플릿 생성 — **순수 함수부터 반드시 테스트**.
- 위젯 테스트는 Settings 화면 값 변경 반영 여부 위주.
- 카메라/TFLite는 mock 서비스로 대체.

---

## 13. Codex 작업 규칙

1. **한 번에 한 기능.** 파일 여러 개를 동시에 갈아엎지 말고, 서비스 하나 → 테스트 → 다음으로 간다.
2. 새 패키지를 추가하면 `pubspec.yaml`과 이 문서의 §1 표를 같이 업데이트한다.
3. 탐지 루프(`detection_service.dart`, `camera_service.dart`)를 수정할 때는 **항상 FPS 영향**을 먼저 따진다. 동기 I/O, Dart 단일 스레드 이미지 연산, 프레임당 객체 할당 금지.
4. 상수 값(임계값, 거리, 쿨다운)을 바꿀 땐 이 문서에 근거가 있는지 확인하고, 없으면 문서에 추가한다.
5. **모르면 추측하지 말고 물어본다.** 특히 ARCore Platform Channel, 기기별 FoV, AI Hub 데이터 구조는 실측/확인이 필요하다.
6. 접근성 관련 코드(`Semantics`, 터치 영역, TTS 안내)는 "나중에"로 미루지 않는다. 이 앱의 사용자는 화면을 못 본다.
7. **확정된 계획과 인터페이스 계약을 임의로 바꾸지 않는다.** ⚠️ 가장 중요

   프론트(Flutter)·백엔드(Spring Boot) 담당자가 이 문서와 `docs/MODEL_INTERFACE.md`를
   **계약으로 삼고 각자 코딩하고 있다.** 스펙이 조용히 바뀌면 그들이 이미 짠 코드가 어긋나고,
   바뀐 줄 모른 채 디버깅하게 된다.

   **바꾸면 안 되는 것 (변경이 필요하면 먼저 물어볼 것):**
   - 15클래스 목록·순서·id (`configs/class_labels.yaml` → `configs/labels.txt`)
   - 모델 입출력 계약 — 입력 `[1,640,640,3]` NHWC/RGB/letterbox,
     출력 `[1,19,8400]`, bbox 0~1 정규화, NMS는 모델 밖, 클래스별 conf 임계값
   - 양자화 방식(INT8), 모델 파일명
   - 백엔드 API 경로·요청/응답 스키마, DB 스키마 (§9)
   - 이 문서에 근거가 적힌 상수값 (임계값·거리·쿨다운)

   **막혔을 때 순서:** ① 원인 규명 → ② 보고 → ③ 선택지 제시 → ④ 결정 대기.
   "이게 더 나으니까"로 대체안을 먼저 실행하지 않는다.

   **판단 기준:** *다른 사람이 문서만 보고 짠 코드가 깨지는가?*
   깨지지 않는 버그 수정·성능 개선·내부 구현 변경은 그대로 진행해도 된다.
   데이터를 보강해 안 잡히던 클래스가 잡히게 되는 것은 계약 변경이 아니다(출력 채널 19 유지).
   단, 동작이 달라지므로 프론트에 알린다.

   불가피하게 바꿨다면 §15(설계 변경 이력)에 기록하고 프론트·백엔드에 공지한다.

   > 실제 사고 (2026-09-27): full-integer INT8 변환이 여러 경로에서 실패하자
   > FP16으로 바꾸고 "§7.4가 fallback으로 규정했으니 계획 밖이 아니다"라고 정당화했다.
   > 실패를 보고하고 결정을 받았어야 했다.

---

## 14. 알려진 리스크 / 미해결

| 항목 | 내용 | 대응 |
|---|---|---|
| ARCore Depth Flutter 연동 | 공식 Flutter 플러그인 없음 → Kotlin Platform Channel 직접 구현 필요 | Phase 3으로 미룸. Phase 2는 bbox 비례식만으로 동작 |
| 기기별 카메라 FoV | `camera` 플러그인이 FoV를 노출하지 않음 | 기본 67° 상수, 추후 Camera2 API로 `LENS_INFO` 조회 |
| YUV420 변환 비용 | Dart 루프로 하면 프레임당 수십 ms | isolate 처리 + 해상도 720p 제한 |
| 신규 클래스 데이터 부족 | kiosk, tactile_paving 등은 COCO에 없음 | AI Hub 데이터 의존 + 직접 촬영 보강 검토 |
| iOS 지원 | ARCore 미지원, ARKit Depth는 LiDAR 기기만 | iOS는 bbox 비례식 경로만 우선 지원 |

---

## 15. 보고서와 달라진 점 (설계 변경 이력)

> 최종보고서 초안 대비 아래 항목이 변경되었다. 보고서 수정 시 반영할 것.

1. **class weight 1.5배 → 오버샘플링 + threshold 조정** (Ultralytics에 per-class weight 옵션 없음)
2. **"backbone 초기 10 epoch freeze" → `freeze=10` (앞 10개 레이어 고정)** (freeze는 epoch가 아니라 레이어 수)
3. **COCO 겹침 "약 절반" → "일부(person, chair 등)"** (실제 겹치는 건 2~3개)
4. **수동 4단계 변환 파이프라인 → `yolo export format=tflite int8=True` 한 줄** (동일 과정을 내부 수행)
5. **Depth Anything v2를 2순위에서 제외 → bbox 비례식이 2순위** (온디바이스 2중 신경망 부담)
6. **OCR/STT/TTS를 클라우드 API → 온디바이스(ML Kit / OS 엔진)** (네트워크 의존 제거, 비용 0)
7. **INT8 delegate 표기: NPU/GPU → NPU/DSP (NNAPI)** (INT8은 GPU delegate 가속 대상 아님)
8. **모델 크기 표기: best.pt 12MB → 6MB** (12MB는 ONNX FP32 기준)

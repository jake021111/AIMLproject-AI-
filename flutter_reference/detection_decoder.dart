// insight — YOLOv8n TFLite 출력 디코딩 참조 구현
//
// 이 파일의 로직은 AI 저장소의 scripts/verify_tflite.py 와 **완전히 동일한 계산**이며,
// 실제 모델·실제 사진으로 검증되었다. 파이썬 결과와 숫자가 맞는지 대조할 수 있다.
//
// 여기 있는 것은 전부 **순수 함수**다. 카메라도 TFLite도 모르기 때문에
// 바로 단위 테스트할 수 있다 (test/detection_decoder_test.dart 참조).
//
// 모델 계약: docs/MODEL_INTERFACE.md
//   입력  [1, 640, 640, 3]  NHWC / RGB / letterbox / float32 (0.0~1.0)
//   출력  [1, 19, 8400]     float32,  19 = 4(bbox) + 15(classes),  **전치되어 있음**
//   NMS 미포함 — 아래 nonMaxSuppression 을 직접 돌려야 한다.
//
// ⚠️ 현재 배포 모델은 입출력이 float32 라 양자화/역양자화 코드가 필요 없다.
//    나중에 int8 입출력 모델로 바뀔 수 있으니 dtype 은 런타임에 확인하는 게 안전하다.

import 'dart:math' as math;
import 'dart:typed_data';

/// 모델 입력 한 변의 크기.
const int kInputSize = 640;

/// 앵커 후보 수 (= 출력 마지막 차원).
const int kNumAnchors = 8400;

/// 클래스 수 (= 출력 채널 19 - 4).
const int kNumClasses = 15;

/// 클래스별 NMS IoU 임계값.
const double kNmsIouThreshold = 0.45;

/// 클래스 id → 신뢰도 임계값.
///
/// veryHigh 우선순위(계단·에스컬레이터·엘리베이터·점자블록)는 미탐이 치명적이라
/// 0.25 로 낮춘다. 나머지는 0.35. (CLAUDE.md §7.2)
const Map<int, double> kConfThresholds = {
  0: 0.35, // chair
  1: 0.35, // table
  2: 0.35, // pillar
  3: 0.35, // trash_can
  4: 0.35, // signage_stand
  5: 0.35, // person
  6: 0.25, // stairs          ← veryHigh
  7: 0.25, // escalator       ← veryHigh
  8: 0.25, // elevator_door   ← veryHigh
  9: 0.35, // door
  10: 0.35, // automatic_door
  11: 0.35, // kiosk
  12: 0.35, // counter
  13: 0.35, // vending_machine
  14: 0.25, // tactile_paving ← veryHigh
};

/// 클래스 순서. assets/labels.txt 와 **반드시 일치**해야 한다.
const List<String> kLabels = [
  'chair',
  'table',
  'pillar',
  'trash_can',
  'signage_stand',
  'person',
  'stairs',
  'escalator',
  'elevator_door',
  'door',
  'automatic_door',
  'kiosk',
  'counter',
  'vending_machine',
  'tactile_paving',
];

/// letterbox 변환 파라미터. 출력 좌표를 원본으로 되돌릴 때 필요하다.
class LetterboxInfo {
  const LetterboxInfo({
    required this.scale,
    required this.padX,
    required this.padY,
    required this.srcWidth,
    required this.srcHeight,
  });

  /// 원본 → letterbox 축소 비율.
  final double scale;

  /// 좌우 패딩(픽셀). 좌/우 각각 이만큼.
  final double padX;

  /// 상하 패딩(픽셀). 상/하 각각 이만큼.
  final double padY;

  final int srcWidth;
  final int srcHeight;

  /// 720p(1280x720) → scale 0.5, padX 0, padY 140 으로 고정이다.
  /// 카메라 해상도가 안 바뀌면 프레임마다 다시 만들 필요 없다.
  static LetterboxInfo compute(int srcWidth, int srcHeight,
      [int size = kInputSize]) {
    final scale = math.min(size / srcWidth, size / srcHeight);
    final newW = (srcWidth * scale).roundToDouble();
    final newH = (srcHeight * scale).roundToDouble();
    return LetterboxInfo(
      scale: scale,
      padX: (size - newW) / 2.0,
      padY: (size - newH) / 2.0,
      srcWidth: srcWidth,
      srcHeight: srcHeight,
    );
  }
}

/// 탐지 결과 하나. 좌표는 **원본 이미지 픽셀** 기준이다.
class Detection {
  const Detection({
    required this.classId,
    required this.score,
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  final int classId;
  final double score;
  final double left;
  final double top;
  final double right;
  final double bottom;

  String get label =>
      classId >= 0 && classId < kLabels.length ? kLabels[classId] : 'unknown';

  double get width => right - left;
  double get height => bottom - top;
  double get centerX => (left + right) / 2.0;
  double get centerY => (top + bottom) / 2.0;

  @override
  String toString() =>
      '$label ${score.toStringAsFixed(3)} '
      '[${left.toStringAsFixed(1)}, ${top.toStringAsFixed(1)}, '
      '${right.toStringAsFixed(1)}, ${bottom.toStringAsFixed(1)}]';
}

/// 모델 출력을 디코딩해 원본 좌표계의 탐지 목록을 만든다.
///
/// [output] 은 `[1, 19, 8400]` 을 평탄화한 것이다 (row-major).
/// 즉 `output[row * 8400 + anchor]` 로 접근한다. 배치 차원 1은 무시해도 된다.
///
/// 반환값은 신뢰도 내림차순이며 NMS까지 끝난 상태다.
List<Detection> decodeOutput(
  Float32List output,
  LetterboxInfo lb, {
  int numClasses = kNumClasses,
  int numAnchors = kNumAnchors,
  Map<int, double> thresholds = kConfThresholds,
  double iouThreshold = kNmsIouThreshold,
}) {
  final candidates = <Detection>[];

  for (var i = 0; i < numAnchors; i++) {
    // 이 앵커에서 가장 점수 높은 클래스를 찾는다.
    // YOLOv8은 objectness 채널이 없다. 4번 행부터가 곧 최종 클래스 점수이고
    // 이미 sigmoid 가 적용된 0~1 값이다. (v5처럼 obj * cls 를 곱하면 안 된다.)
    var bestId = -1;
    var bestScore = 0.0;
    for (var c = 0; c < numClasses; c++) {
      final s = output[(4 + c) * numAnchors + i];
      if (s > bestScore) {
        bestScore = s;
        bestId = c;
      }
    }
    if (bestId < 0) continue;

    // 클래스마다 임계값이 다르다.
    if (bestScore < (thresholds[bestId] ?? 0.35)) continue;

    // bbox 는 letterbox 640 좌표계 기준으로 0~1 정규화되어 있다.
    final cx = output[0 * numAnchors + i] * kInputSize;
    final cy = output[1 * numAnchors + i] * kInputSize;
    final w = output[2 * numAnchors + i] * kInputSize;
    final h = output[3 * numAnchors + i] * kInputSize;

    // letterbox 역변환 → 원본 픽셀.
    final cxSrc = (cx - lb.padX) / lb.scale;
    final cySrc = (cy - lb.padY) / lb.scale;
    final wSrc = w / lb.scale;
    final hSrc = h / lb.scale;

    candidates.add(Detection(
      classId: bestId,
      score: bestScore,
      left: (cxSrc - wSrc / 2).clamp(0.0, lb.srcWidth.toDouble()),
      top: (cySrc - hSrc / 2).clamp(0.0, lb.srcHeight.toDouble()),
      right: (cxSrc + wSrc / 2).clamp(0.0, lb.srcWidth.toDouble()),
      bottom: (cySrc + hSrc / 2).clamp(0.0, lb.srcHeight.toDouble()),
    ));
  }

  return nonMaxSuppression(candidates, iouThreshold: iouThreshold);
}

/// 두 박스의 IoU.
double iou(Detection a, Detection b) {
  final ix1 = math.max(a.left, b.left);
  final iy1 = math.max(a.top, b.top);
  final ix2 = math.min(a.right, b.right);
  final iy2 = math.min(a.bottom, b.bottom);

  final iw = ix2 - ix1;
  final ih = iy2 - iy1;
  if (iw <= 0 || ih <= 0) return 0.0;

  final inter = iw * ih;
  final union = a.width * a.height + b.width * b.height - inter;
  return union <= 0 ? 0.0 : inter / union;
}

/// 클래스별 NMS.
///
/// **서로 다른 클래스의 박스는 겹쳐도 지우지 않는다.** 의자 위에 사람이 있는 경우
/// 둘 다 남아야 하기 때문이다.
List<Detection> nonMaxSuppression(
  List<Detection> candidates, {
  double iouThreshold = kNmsIouThreshold,
}) {
  final sorted = List<Detection>.from(candidates)
    ..sort((a, b) => b.score.compareTo(a.score));

  final kept = <Detection>[];
  for (final cand in sorted) {
    var suppressed = false;
    for (final k in kept) {
      if (k.classId == cand.classId && iou(cand, k) > iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.add(cand);
  }
  return kept;
}

/// 화면 중심 기준 각도(도). 좌측이 음수, 우측이 양수.
///
/// CLAUDE.md §6.4 의 선형 근사식이다. 안내 용도로는 충분하다.
double angleDegrees(Detection d, int imageWidth, {double fovDeg = 67.0}) {
  return (d.centerX / imageWidth - 0.5) * fovDeg;
}

/// 각도를 시계 방향으로 변환한다. 12시가 정면.
int toClockDirection(double angleDeg) {
  if (angleDeg < -20) return 10;
  if (angleDeg < -7) return 11;
  if (angleDeg <= 7) return 12;
  if (angleDeg <= 20) return 1;
  return 2;
}

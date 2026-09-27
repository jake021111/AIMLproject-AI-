// insight — 거리 추정 참조 구현 (CLAUDE.md §6.2)
//
// 모델은 거리를 출력하지 않는다. 출력은 [1, 19, 8400] = bbox 4 + 클래스 15 가 전부다.
// 거리는 여기서 **bbox 크기로부터 계산**한다. 학습도 추론도 아니고 닮은꼴 삼각형 비례식이다.
//
//     거리(m) = (그 클래스의 평균 실측 폭 × 초점거리) / bbox 픽셀 폭
//
// Phase 3 에서 ARCore Depth 가 붙으면 이 비례식은 2순위 fallback 이 된다 (§6.1).
// 그래서 DistanceSource 로 어느 방식으로 구한 값인지 남겨 둔다.

import 'dart:math' as math;

import 'detection_decoder.dart';

/// 거리 값을 어디서 얻었는지.
enum DistanceSource {
  /// ARCore Depth API (Phase 3). 절대 거리, 정확.
  arcore,

  /// bbox 비례식. 근사값이라 "약 ○미터" 로 안내해야 한다.
  bboxRatio,

  /// 추정 실패 또는 거리 안내를 하지 않는 클래스.
  none,
}

/// 클래스별 평균 실측 폭(m). null 이면 거리 추정을 하지 않는다.
///
/// `stairs`(6)와 `tactile_paving`(14)은 바닥에 깔린 넓은 영역이라
/// bbox 폭과 실제 폭이 대응하지 않는다. 거리 수치 대신
/// "전방에 계단이 있습니다" 형태로만 안내한다. (CLAUDE.md §3.1)
///
/// 원본: AI 저장소 configs/class_labels.yaml
const Map<int, double?> kRealWidthM = {
  0: 0.50, // chair
  1: 1.20, // table
  2: 0.50, // pillar
  3: 0.40, // trash_can
  4: 0.60, // signage_stand
  5: 0.50, // person
  6: null, // stairs          ← 거리 추정 안 함
  7: 1.00, // escalator
  8: 1.00, // elevator_door
  9: 0.90, // door
  10: 1.20, // automatic_door
  11: 0.60, // kiosk
  12: 1.50, // counter
  13: 1.00, // vending_machine
  14: null, // tactile_paving  ← 거리 추정 안 함
};

/// 기기 카메라 수평 화각(도).
///
/// `camera` 플러그인이 FoV 를 노출하지 않아 기기별로 알 수 없다.
/// 기본 67° 로 두고 상수화한다 (CLAUDE.md §6.2, §14).
/// 나중에 Camera2 API 의 LENS_INFO 로 실측하면 정확도가 올라간다.
const double kDefaultCameraFovDeg = 67.0;

/// 거리 추정 결과.
class DistanceEstimate {
  const DistanceEstimate({
    required this.distanceM,
    required this.source,
  });

  /// 추정 거리(m). null 이면 거리를 말하지 않는다.
  final double? distanceM;

  final DistanceSource source;

  /// 거리 수치를 음성으로 말해도 되는지.
  bool get hasDistance => distanceM != null;

  /// bbox 비례식은 오차가 ±30% 수준이라 "약" 을 붙여야 한다 (§6.2).
  bool get isApproximate => source == DistanceSource.bboxRatio;

  static const none =
      DistanceEstimate(distanceM: null, source: DistanceSource.none);
}

double _toRad(double deg) => deg * math.pi / 180.0;

/// 초점거리(픽셀)를 구한다.
///
/// 카메라 해상도와 화각이 안 바뀌면 값도 안 바뀐다. **프레임마다 다시 계산하지 말고**
/// 한 번 구해서 들고 있을 것 (탐지 루프에서 삼각함수는 낭비다 — CLAUDE.md §5.2).
double focalLengthPx(int imageWidth, {double fovDeg = kDefaultCameraFovDeg}) {
  return (imageWidth / 2) / math.tan(_toRad(fovDeg) / 2);
}

/// bbox 비례식으로 거리를 추정한다.
///
/// [focalPx] 는 [focalLengthPx] 로 미리 구해 둔 값이다.
DistanceEstimate estimateDistance(Detection d, double focalPx) {
  final realWidth = kRealWidthM[d.classId];

  // 거리 안내를 하지 않는 클래스 (계단, 점자블록).
  if (realWidth == null) return DistanceEstimate.none;

  // bbox 가 비정상적으로 작으면 나눗셈이 폭주한다.
  if (d.width < 1.0) return DistanceEstimate.none;

  final distance = (realWidth * focalPx) / d.width;

  // 실내 보행에서 의미 없는 범위는 버린다.
  // 0.3m 미만이면 이미 부딪힌 것이고, 20m 초과는 안내할 가치가 없다.
  if (distance < 0.3 || distance > 20.0) return DistanceEstimate.none;

  return DistanceEstimate(
    distanceM: distance,
    source: DistanceSource.bboxRatio,
  );
}

/// 음성 안내 문장을 만든다 (CLAUDE.md §8.2).
///
/// 순서는 **방향 → 거리 → 객체**. 색상·재질·배경 묘사는 하지 않는다.
///
/// 예)
///   "11시 방향 약 2미터, 의자"
///   "전방에 계단이 있습니다"
///   "정면 약 1미터, 사람. 주의하세요"
String buildSpeech(
  Detection d,
  DistanceEstimate dist, {
  required String koreanLabel,
  required int clockDirection,
  bool isWarning = false,
}) {
  final direction = clockDirection == 12 ? '정면' : '$clockDirection시 방향';

  final String body;
  if (!dist.hasDistance) {
    // 거리를 모르거나 말하면 안 되는 클래스.
    body = '전방에 $koreanLabel 있습니다';
  } else {
    final meters = dist.distanceM!.round();
    final approx = dist.isApproximate ? '약 ' : '';
    body = '$direction $approx$meters미터, $koreanLabel';
  }

  return isWarning ? '$body. 주의하세요' : body;
}

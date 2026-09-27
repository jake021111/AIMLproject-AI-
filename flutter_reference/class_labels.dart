// 자동 생성 파일 — 직접 수정하지 마세요.
//
// 생성: python scripts/gen_class_labels_dart.py
// 원본: AI 저장소 configs/class_labels.yaml
//
// 클래스 정의의 단일 소스는 yaml 이다. 여기를 손으로 고치면 모델이 내는 class_id 와
// 앱이 말하는 라벨이 어긋난다 (CLAUDE.md §3.1, §13 규칙 2).
//
// 모델 출력의 class_id 가 곧 이 리스트의 인덱스다.

/// 안내 우선순위. veryHigh 는 미탐(FN)이 치명적이라 임계값을 낮춘다.
enum GuidePriority { veryHigh, high, medium }

class ClassInfo {
  const ClassInfo({
    required this.id,
    required this.label,
    required this.korean,
    required this.priority,
    required this.realWidthM,
    required this.distanceReliable,
    required this.confThreshold,
  });

  /// 모델 출력의 class_id.
  final int id;

  /// 모델 라벨 (영문). assets/labels.txt 와 일치한다.
  final String label;

  /// TTS 출력용 한국어 이름.
  final String korean;

  final GuidePriority priority;

  /// bbox 거리 추정용 평균 실측 폭(m). null 이면 거리 추정을 하지 않는다.
  final double? realWidthM;

  /// false 면 거리 수치 대신 "전방에 ○○ 있습니다" 로만 안내한다.
  final bool distanceReliable;

  /// 이 클래스의 신뢰도 임계값.
  final double confThreshold;
}


/// 15개 클래스 정의. **인덱스 = class_id** 이므로 순서를 바꾸면 안 된다.
const List<ClassInfo> kClassInfos = [
  ClassInfo(
    id: 0,
    label: 'chair',
    korean: '의자',
    priority: GuidePriority.high,
    realWidthM: 0.50,
    distanceReliable: true,
    confThreshold: 0.35,
  ),
  ClassInfo(
    id: 1,
    label: 'table',
    korean: '테이블',
    priority: GuidePriority.high,
    realWidthM: 1.20,
    distanceReliable: true,
    confThreshold: 0.35,
  ),
  ClassInfo(
    id: 2,
    label: 'pillar',
    korean: '기둥',
    priority: GuidePriority.high,
    realWidthM: 0.50,
    distanceReliable: true,
    confThreshold: 0.35,
  ),
  ClassInfo(
    id: 3,
    label: 'trash_can',
    korean: '쓰레기통',
    priority: GuidePriority.high,
    realWidthM: 0.40,
    distanceReliable: true,
    confThreshold: 0.35,
  ),
  ClassInfo(
    id: 4,
    label: 'signage_stand',
    korean: '안내 표지판',
    priority: GuidePriority.high,
    realWidthM: 0.60,
    distanceReliable: true,
    confThreshold: 0.35,
  ),
  ClassInfo(
    id: 5,
    label: 'person',
    korean: '사람',
    priority: GuidePriority.high,
    realWidthM: 0.50,
    distanceReliable: true,
    confThreshold: 0.35,
  ),
  ClassInfo(
    id: 6,
    label: 'stairs',
    korean: '계단',
    priority: GuidePriority.veryHigh,
    realWidthM: null,
    distanceReliable: false,
    confThreshold: 0.25,
  ),
  ClassInfo(
    id: 7,
    label: 'escalator',
    korean: '에스컬레이터',
    priority: GuidePriority.veryHigh,
    realWidthM: 1.00,
    distanceReliable: true,
    confThreshold: 0.25,
  ),
  ClassInfo(
    id: 8,
    label: 'elevator_door',
    korean: '엘리베이터',
    priority: GuidePriority.veryHigh,
    realWidthM: 1.00,
    distanceReliable: true,
    confThreshold: 0.25,
  ),
  ClassInfo(
    id: 9,
    label: 'door',
    korean: '문',
    priority: GuidePriority.high,
    realWidthM: 0.90,
    distanceReliable: true,
    confThreshold: 0.35,
  ),
  ClassInfo(
    id: 10,
    label: 'automatic_door',
    korean: '자동문',
    priority: GuidePriority.high,
    realWidthM: 1.20,
    distanceReliable: true,
    confThreshold: 0.35,
  ),
  ClassInfo(
    id: 11,
    label: 'kiosk',
    korean: '키오스크',
    priority: GuidePriority.medium,
    realWidthM: 0.60,
    distanceReliable: true,
    confThreshold: 0.35,
  ),
  ClassInfo(
    id: 12,
    label: 'counter',
    korean: '카운터',
    priority: GuidePriority.medium,
    realWidthM: 1.50,
    distanceReliable: true,
    confThreshold: 0.35,
  ),
  ClassInfo(
    id: 13,
    label: 'vending_machine',
    korean: '자판기',
    priority: GuidePriority.medium,
    realWidthM: 1.00,
    distanceReliable: true,
    confThreshold: 0.35,
  ),
  ClassInfo(
    id: 14,
    label: 'tactile_paving',
    korean: '점자블록',
    priority: GuidePriority.veryHigh,
    realWidthM: null,
    distanceReliable: false,
    confThreshold: 0.25,
  ),
];

/// class_id 로 클래스 정보를 얻는다. 범위를 벗어나면 null.
ClassInfo? classInfoOf(int classId) =>
    (classId >= 0 && classId < kClassInfos.length)
        ? kClassInfos[classId]
        : null;

/// TTS 용 한국어 이름. 모르는 id 면 '물체'.
String koreanOf(int classId) => classInfoOf(classId)?.korean ?? '물체';

/// 미탐이 치명적인 클래스들: stairs, escalator, elevator_door, tactile_paving
bool isVeryHigh(int classId) =>
    classInfoOf(classId)?.priority == GuidePriority.veryHigh;

"""configs/class_labels.yaml -> Flutter 용 class_labels.dart 생성.

CLAUDE.md §3.1 의 ClassInfo 를 그대로 Dart 로 옮긴다.
클래스 정의의 단일 소스는 yaml 이고, Dart 파일은 **항상 여기서 생성**한다.
손으로 고치면 두 파일이 어긋나서 앱이 엉뚱한 라벨을 말하게 된다 (CLAUDE.md §13 규칙 2).

사용법:
    python scripts/gen_class_labels_dart.py
"""
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "configs" / "class_labels.yaml"
DST = ROOT / "flutter_reference" / "class_labels.dart"

PRIORITY_DART = {
    "very_high": "GuidePriority.veryHigh",
    "high": "GuidePriority.high",
    "medium": "GuidePriority.medium",
}

HEADER = """// 자동 생성 파일 — 직접 수정하지 마세요.
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

"""


def main() -> None:
    cfg = yaml.safe_load(SRC.read_text(encoding="utf-8"))
    classes = sorted(cfg["classes"], key=lambda c: c["id"])

    expected = list(range(len(classes)))
    if [c["id"] for c in classes] != expected:
        raise ValueError(f"class id 가 0..{len(classes) - 1} 연속이 아닙니다.")

    lines = [HEADER]
    lines.append("/// 15개 클래스 정의. **인덱스 = class_id** 이므로 순서를 바꾸면 안 된다.")
    lines.append("const List<ClassInfo> kClassInfos = [")
    for c in classes:
        rw = "null" if c["real_width_m"] is None else f"{float(c['real_width_m']):.2f}"
        lines.append(
            f"""  ClassInfo(
    id: {c['id']},
    label: '{c['label']}',
    korean: '{c['korean']}',
    priority: {PRIORITY_DART[c['priority']]},
    realWidthM: {rw},
    distanceReliable: {str(c['distance_reliable']).lower()},
    confThreshold: {float(c['conf_threshold'])},
  ),"""
        )
    lines.append("];")
    lines.append("")
    lines.append("/// class_id 로 클래스 정보를 얻는다. 범위를 벗어나면 null.")
    lines.append("ClassInfo? classInfoOf(int classId) =>")
    lines.append("    (classId >= 0 && classId < kClassInfos.length)")
    lines.append("        ? kClassInfos[classId]")
    lines.append("        : null;")
    lines.append("")
    lines.append("/// TTS 용 한국어 이름. 모르는 id 면 '물체'.")
    lines.append("String koreanOf(int classId) => classInfoOf(classId)?.korean ?? '물체';")
    lines.append("")
    vh = [c["label"] for c in classes if c["priority"] == "very_high"]
    lines.append(f"/// 미탐이 치명적인 클래스들: {', '.join(vh)}")
    lines.append("bool isVeryHigh(int classId) =>")
    lines.append("    classInfoOf(classId)?.priority == GuidePriority.veryHigh;")
    lines.append("")

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text("\n".join(lines), encoding="utf-8")
    print(f"{DST.relative_to(ROOT)} 생성 완료 ({len(classes)} classes)")
    for c in classes:
        print(f"  {c['id']:2d} {c['label']:16s} {c['korean']}")


if __name__ == "__main__":
    main()

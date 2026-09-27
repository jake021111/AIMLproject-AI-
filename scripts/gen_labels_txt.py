"""configs/class_labels.yaml -> configs/labels.txt 생성.

Flutter 앱 assets/labels.txt 와 동일한 파일이며, 클래스 순서(= 모델 출력 채널 순서)의
단일 소스는 configs/class_labels.yaml 이다. 손으로 labels.txt 를 고치지 말고 이 스크립트를 돌린다.

사용법:
    python scripts/gen_labels_txt.py
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "configs" / "class_labels.yaml"
DST = ROOT / "configs" / "labels.txt"


def main() -> None:
    classes = yaml.safe_load(SRC.read_text(encoding="utf-8"))["classes"]
    classes.sort(key=lambda c: c["id"])

    # id 가 0..N-1 연속인지 검증 — 어긋나면 앱 파싱이 전부 밀린다.
    expected = list(range(len(classes)))
    actual = [c["id"] for c in classes]
    if actual != expected:
        raise ValueError(f"class id 가 0..{len(classes) - 1} 연속이 아닙니다: {actual}")

    DST.write_text("\n".join(c["label"] for c in classes) + "\n", encoding="utf-8")
    print(f"{DST.relative_to(ROOT)} 생성 완료 ({len(classes)} classes)")


if __name__ == "__main__":
    main()

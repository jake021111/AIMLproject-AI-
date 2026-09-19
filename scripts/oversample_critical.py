"""veryHigh 우선순위 클래스(stairs, escalator, elevator_door, tactile_paving)가
포함된 학습 이미지를 오버샘플링(기본 2배 복제)한다.

Ultralytics 기본 파이프라인은 per-class loss weight를 지원하지 않으므로
(CLAUDE.md §7.2, §15.1 — cls loss가 BCE라 직접 넣으려면 loss 코드 수정이 필요함),
데이터셋 구성 단계에서 인스턴스 수를 늘려 recall을 확보하는 방식을 쓴다.
train split에만 적용할 것 (val/test에 적용하면 지표가 왜곡된다).

사용법:
    python scripts/oversample_critical.py --images dataset/images/train --labels dataset/labels/train
"""
import argparse
import shutil
from pathlib import Path

VERY_HIGH_IDS = {6, 7, 8, 14}  # stairs, escalator, elevator_door, tactile_paving
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="veryHigh 클래스 오버샘플링 (train split 전용)")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--factor", type=int, default=2, help="총 배수 (기본 2배 = 원본 1 + 복제 1)")
    return parser.parse_args()


def label_has_very_high(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id = int(line.split()[0])
        if class_id in VERY_HIGH_IDS:
            return True
    return False


def main() -> None:
    args = parse_args()
    copies_needed = args.factor - 1
    if copies_needed <= 0:
        print("factor는 2 이상이어야 합니다.")
        return

    targets = [
        img
        for img in sorted(args.images.iterdir())
        if img.suffix.lower() in IMG_EXTS and label_has_very_high(args.labels / f"{img.stem}.txt")
    ]
    print(f"오버샘플링 대상: {len(targets)}장 (x{args.factor})")

    for img_path in targets:
        label_path = args.labels / f"{img_path.stem}.txt"
        for copy_idx in range(1, copies_needed + 1):
            new_img = args.images / f"{img_path.stem}_vh{copy_idx}{img_path.suffix}"
            new_label = args.labels / f"{img_path.stem}_vh{copy_idx}.txt"
            if new_img.exists():
                continue
            shutil.copy2(img_path, new_img)
            shutil.copy2(label_path, new_label)

    print("완료.")


if __name__ == "__main__":
    main()

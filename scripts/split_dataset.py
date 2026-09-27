"""평탄한 images/ + labels/ 디렉터리를 train/val/test (기본 70/20/10)로 분할한다.

YOLO 포맷 전제: 이미지(jpg/png)마다 같은 이름의 .txt 라벨 파일이 존재해야 한다.
라벨이 없는 이미지는 건너뛰고 경고를 출력한다.

사용법:
    python scripts/split_dataset.py --src raw_dataset --dst dataset
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train/val/test 분할 (기본 70/20/10)")
    parser.add_argument("--src", required=True, type=Path, help="images/, labels/ 하위 폴더를 가진 원본 경로")
    parser.add_argument("--dst", required=True, type=Path, help="분할 결과 저장 경로 (data.yaml의 path)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preserve-from",
        type=Path,
        default=None,
        help="기존 분할 결과 경로(images/{train,val,test} 구조). 여기에 이미 있는 이미지는 "
             "같은 split 에 그대로 둔다. 데이터를 추가하고 재학습할 때 예전 test 이미지가 "
             "새 train 으로 새어 들어가는 것을 막아, 이전 모델과 지표를 공정하게 비교할 수 있다.",
    )
    return parser.parse_args()


def load_previous_split(path: Path) -> dict[str, str]:
    """기존 분할에서 stem -> split 맵을 만든다."""
    mapping: dict[str, str] = {}
    for split in ("train", "val", "test"):
        d = path / "images" / split
        if not d.is_dir():
            continue
        for img in d.iterdir():
            if img.suffix.lower() in IMG_EXTS:
                mapping[img.stem] = split
    return mapping


def match_previous(stem: str, previous: dict[str, str]) -> str | None:
    """merge_sources.py 가 붙인 '<접두사>_' 를 벗겨 가며 기존 split 을 찾는다."""
    if stem in previous:
        return previous[stem]
    if "_" in stem:
        return previous.get(stem.split("_", 1)[1])
    return None


def main() -> None:
    args = parse_args()
    src_images = args.src / "images"
    src_labels = args.src / "labels"

    images = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"이미지가 없습니다: {src_images}")

    previous = load_previous_split(args.preserve_from) if args.preserve_from else {}
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    fresh = []
    kept = 0
    for img in images:
        assigned = match_previous(img.stem, previous) if previous else None
        if assigned:
            splits[assigned].append(img)
            kept += 1
        else:
            fresh.append(img)

    if previous:
        print(f"기존 분할 유지: {kept}장 / 새로 분할: {len(fresh)}장")

    random.seed(args.seed)
    random.shuffle(fresh)

    n_train = int(len(fresh) * args.train_ratio)
    n_val = int(len(fresh) * args.val_ratio)
    splits["train"] += fresh[:n_train]
    splits["val"] += fresh[n_train : n_train + n_val]
    splits["test"] += fresh[n_train + n_val :]

    for split_name, split_images in splits.items():
        img_dir = args.dst / "images" / split_name
        lbl_dir = args.dst / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        skipped = 0
        for img_path in split_images:
            label_path = src_labels / f"{img_path.stem}.txt"
            if not label_path.exists():
                skipped += 1
                continue
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(label_path, lbl_dir / label_path.name)

        print(f"{split_name}: {len(split_images) - skipped}장 복사 (라벨 없어 건너뜀 {skipped}장)")


if __name__ == "__main__":
    main()

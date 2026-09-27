"""여러 출처(AI Hub / ADE20K / COCO)의 YOLO 데이터를 하나로 합친다.

각 출처는 `images/` + `labels/` 구조를 가진 폴더다.
파일명이 겹치지 않도록 출처별 접두사를 붙이고, 출처별 분량 상한을 둘 수 있다.

왜 분량을 통제하나:
    우리 목표 도메인은 **한국 지하철·공공건물 실내**(AI Hub 186)다.
    ADE20K(서구권 가정·사무실)와 COCO(일반 사진)를 무제한으로 넣으면
    도메인이 흐려져 정작 목표 환경에서 성능이 떨어질 수 있다.
    또 보조 출처 이미지에는 우리 클래스가 찍혀 있어도 라벨이 없는 경우가 있어
    (예: COCO 사진 속 문·기둥) 그만큼 '배경'으로 학습된다.
    -> AI Hub 를 주(主)로 두고 보조 출처는 제한적으로 섞는다.

사용법:
    python scripts/merge_sources.py \\
        --src raw_dataset:aihub \\
        --src ade20k_subset:ade:1200 \\
        --src coco_subset:coco:800 \\
        --out merged_dataset

    # 형식: 경로:접두사[:최대이미지수]

    이어서:
    python scripts/split_dataset.py --src merged_dataset --dst dataset
"""
import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="여러 출처의 YOLO 데이터 병합")
    parser.add_argument("--src", action="append", required=True,
                        help="경로:접두사[:최대이미지수] 형식. 여러 번 지정 가능.")
    parser.add_argument("--out", type=Path, default=ROOT / "merged_dataset")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_names() -> dict[int, str]:
    cfg = yaml.safe_load((ROOT / "configs" / "class_labels.yaml").read_text(encoding="utf-8"))
    return {c["id"]: c["label"] for c in cfg["classes"]}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    names = load_names()

    images_out = args.out / "images"
    labels_out = args.out / "labels"
    if args.out.exists():
        shutil.rmtree(args.out)
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    per_source: dict[str, Counter] = {}
    per_source_imgs: dict[str, int] = {}

    for spec in args.src:
        parts = spec.split(":")
        if len(parts) < 2:
            raise SystemExit(f"--src 형식이 잘못됐습니다: {spec} (경로:접두사[:최대수])")
        src_path = Path(parts[0])
        prefix = parts[1]
        limit = int(parts[2]) if len(parts) > 2 and parts[2] else 0

        src_images = src_path / "images"
        src_labels = src_path / "labels"
        if not src_images.is_dir():
            raise SystemExit(f"images 폴더가 없습니다: {src_images}")

        pairs = []
        for img in sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMG_EXTS):
            lbl = src_labels / f"{img.stem}.txt"
            if lbl.exists() and lbl.stat().st_size > 0:
                pairs.append((img, lbl))

        if limit and len(pairs) > limit:
            random.shuffle(pairs)
            pairs = pairs[:limit]

        counter = Counter()
        for img, lbl in pairs:
            stem = f"{prefix}_{img.stem}"
            shutil.copy2(img, images_out / f"{stem}{img.suffix.lower()}")
            shutil.copy2(lbl, labels_out / f"{stem}.txt")
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    counter[int(line.split()[0])] += 1

        per_source[prefix] = counter
        per_source_imgs[prefix] = len(pairs)
        print(f"{prefix:8s} {len(pairs):5d}장  {sum(counter.values()):6d}박스  <- {src_path}")

    total = Counter()
    for c in per_source.values():
        total.update(c)

    print(f"\n합계: {sum(per_source_imgs.values())}장 / {sum(total.values())}박스 -> {args.out}")
    print("\n클래스별 박스 수 (출처별 내역)")
    header = "  id 클래스            " + "".join(f"{p:>9s}" for p in per_source) + "     합계"
    print(header)
    for cid in sorted(names):
        row = "".join(f"{per_source[p].get(cid, 0):9d}" for p in per_source)
        mark = "  <- 없음" if total.get(cid, 0) == 0 else ""
        print(f"  {cid:2d} {names[cid]:16s}{row} {total.get(cid, 0):8d}{mark}")

    print("\n다음: python scripts/split_dataset.py --src "
          f"{args.out.name} --dst dataset")


if __name__ == "__main__":
    main()

"""AI Hub 186 "베리어프리존 주행영상" 원본 라벨(JSON)을 YOLO txt 포맷으로 변환한다.

원본 구조 (실측 확인됨, 2026-09-20):
    <root>/Training|Validation/[원천]<지역명>/  *.jpg
    <root>/Training|Validation/[라벨]<지역명>/  *.json          (실제 bbox/polygon 라벨)
    <root>/Training|Validation/[라벨]<지역명>_s/ *_s.json        (촬영 메타데이터 — GPS/날씨 등,
                                                                  객체 좌표 없음 → 사용 안 함)

라벨 JSON 스키마:
    {"images": {"file_name": ..., "width": ..., "height": ...},
     "annotations": [{"id": "rectangle_N"|"polygon_N", "category_id": <str>,
                       "segmentation": [[x,y], ...]}]}
    - id가 "rectangle_"로 시작 -> segmentation은 대각 꼭짓점 2개 [ [x1,y1], [x2,y2] ]
    - id가 "polygon_"으로 시작  -> segmentation은 다각형 꼭짓점 목록 -> min/max로 bbox 변환

AI Hub 자체 Training/Validation 분할은 쓰지 않는다. 여기서는 전부 하나의 풀로 모으고,
scripts/split_dataset.py가 프로젝트 규칙(70/20/10, CLAUDE.md §7.1)대로 다시 나눈다.

사용법:
    python scripts/prepare_dataset.py \
        --root "data folder/베리어프리존(장애물 없는 생활공간) 주행영상" \
        --label-map label_map.yaml \
        --out raw_dataset
"""
import argparse
import json
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Hub 186 라벨을 YOLO 포맷으로 변환")
    parser.add_argument("--root", required=True, type=Path, help="AI Hub 압축 해제 루트 (Training/Validation을 담은 폴더)")
    parser.add_argument("--label-map", default="label_map.yaml", type=Path)
    parser.add_argument("--out", default="raw_dataset", type=Path, help="images/, labels/ 를 만들 출력 경로")
    parser.add_argument("--keep-empty", action="store_true", help="매핑되는 객체가 하나도 없는 이미지도 포함(빈 라벨)")
    return parser.parse_args()


def load_category_map(label_map_path: Path) -> dict[str, int]:
    with open(label_map_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    target_classes: dict[int, str] = data["target_classes"]
    name_to_id = {name: cid for cid, name in target_classes.items()}

    aihub_map: dict[str, int] = {}
    for category_id, target_name in data.get("aihub_186", {}).items():
        aihub_map[category_id] = name_to_id[target_name]
    return aihub_map


def bbox_from_segmentation(points: list[list[float]], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = max(0.0, min(xs))
    x_max = min(float(img_w), max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(float(img_h), max(ys))

    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return cx, cy, w, h


def find_region_pairs(root: Path) -> list[tuple[Path, Path]]:
    """(원천 이미지 폴더, 라벨 폴더) 쌍을 찾는다. 같은 split(Training/Validation) 안에서
    폴더명이 '[원천]<지역명>' / '[라벨]<지역명>' 으로 정확히 대응하는 것만 짝짓는다.
    (label _s 폴더는 접미사가 붙어 이름이 달라 자동으로 제외됨)"""
    pairs = []
    for split_dir in root.iterdir():
        if not split_dir.is_dir():
            continue
        source_dirs = {d.name.removeprefix("[원천]"): d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("[원천]")}
        label_dirs = {d.name.removeprefix("[라벨]"): d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("[라벨]")}

        for region, source_dir in source_dirs.items():
            label_dir = label_dirs.get(region)
            if label_dir is None:
                print(f"경고: '{region}' 원천은 있는데 매칭되는 라벨 폴더가 없음 -> 건너뜀")
                continue
            pairs.append((source_dir, label_dir))
    return pairs


def main() -> None:
    args = parse_args()
    category_map = load_category_map(args.label_map)

    out_images = args.out / "images"
    out_labels = args.out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    pairs = find_region_pairs(args.root)
    if not pairs:
        raise FileNotFoundError(f"'{args.root}' 아래에서 [원천]/[라벨] 폴더 쌍을 찾지 못했습니다.")

    total_images = 0
    total_boxes = 0
    skipped_no_image = 0
    skipped_empty = 0
    class_counts: dict[int, int] = {}

    for source_dir, label_dir in pairs:
        prefix = label_dir.name.removeprefix("[라벨]")
        label_files = [p for p in label_dir.iterdir() if p.suffix == ".json" and not p.stem.endswith("_s")]

        for label_path in label_files:
            with open(label_path, encoding="utf-8") as f:
                data = json.load(f)

            file_name = data["images"]["file_name"]
            img_w = data["images"]["width"]
            img_h = data["images"]["height"]
            src_image = source_dir / file_name
            if not src_image.exists():
                skipped_no_image += 1
                continue

            lines = []
            for ann in data.get("annotations", []):
                class_id = category_map.get(ann["category_id"])
                if class_id is None:
                    continue  # 15-class 체계와 무관한 카테고리 (wall, floor 등)
                cx, cy, w, h = bbox_from_segmentation(ann["segmentation"], img_w, img_h)
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                total_boxes += 1

            if not lines and not args.keep_empty:
                skipped_empty += 1
                continue

            out_name = f"{prefix}__{src_image.stem}"
            dst_image = out_images / f"{out_name}{src_image.suffix}"
            dst_label = out_labels / f"{out_name}.txt"
            dst_image.write_bytes(src_image.read_bytes())
            dst_label.write_text("\n".join(lines), encoding="utf-8")
            total_images += 1

    print(f"\n변환 완료: 이미지 {total_images}장, 박스 {total_boxes}개")
    print(f"건너뜀: 원본 이미지 없음 {skipped_no_image}장, 매핑 객체 없음(빈 라벨) {skipped_empty}장")
    print("클래스별 박스 개수:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  class {class_id}: {count}")


if __name__ == "__main__":
    main()

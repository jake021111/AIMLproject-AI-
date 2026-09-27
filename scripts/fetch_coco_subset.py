"""COCO 2017 에서 person / chair / dining table 만 뽑아 YOLO 포맷으로 변환한다.

AI Hub 186 데이터셋에는 person, table 이 아예 없고 chair 도 부족하다(train 71개).
COCO 에서 해당 클래스만 보강한다 (CLAUDE.md §7.1 "겹치는 클래스만 선별").

COCO 전체(train2017 18GB)를 받지 않는다. 어노테이션(241MB)만 받아서 필요한 이미지의
coco_url 로 **선택한 장수만** 내려받는다.

클래스 매핑 (COCO category_id -> insight class id):
    1  person       -> 5  person
    62 chair        -> 0  chair
    67 dining table -> 1  table

주의 — 배경 오염(background contamination):
    COCO 이미지에는 door, pillar, trash_can 같은 우리 클래스가 찍혀 있어도 라벨이 없다.
    모델은 그걸 "배경"으로 학습하게 되므로 AI Hub 쪽 클래스의 recall 이 떨어질 수 있다.
    그래서 COCO 분량을 AI Hub 대비 과하게 넣지 않는다 (--max-per-class 기본 600).
    실내 위주로 거르려면 --indoor-only 를 쓴다 (실내 계열 supercategory 가 같이 찍힌 것만 채택).

사용법:
    python scripts/fetch_coco_subset.py --out coco_subset
    python scripts/fetch_coco_subset.py --out coco_subset --max-per-class 400 --indoor-only

결과:
    coco_subset/images/*.jpg, coco_subset/labels/*.txt  (YOLO 포맷)
    -> scripts/split_dataset.py 로 분할하거나, dataset/ 에 직접 합쳐 쓴다.
"""
import argparse
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# COCO category_id -> insight class id
COCO_TO_INSIGHT = {1: 5, 62: 0, 67: 1}
INSIGHT_NAMES = {5: "person", 0: "chair", 1: "table"}

# --indoor-only 에서 "실내 장면"으로 인정할 동반 객체들 (COCO category_id).
# 가구·주방·전자기기 계열이 같이 찍혀 있으면 실내로 본다.
INDOOR_HINTS = {
    62, 63, 64, 65, 67, 70, 72, 73, 78, 79, 82, 84, 85, 86,
    44, 46, 47, 48, 49, 50, 51,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO 2017 부분 추출 -> YOLO 포맷")
    parser.add_argument("--out", type=Path, default=ROOT / "coco_subset")
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=ROOT / "coco_annotations",
        help="어노테이션 zip 을 받아 풀어둘 경로 (재실행 시 재사용)",
    )
    parser.add_argument(
        "--split", default="train2017", choices=["train2017", "val2017"]
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=600,
        help="클래스당 최대 이미지 수 (배경 오염 억제를 위해 과하게 넣지 않는다)",
    )
    parser.add_argument("--min-box-area", type=float, default=0.005,
                        help="이미지 면적 대비 최소 박스 비율 (너무 작은 객체 제외)")
    parser.add_argument("--indoor-only", action="store_true",
                        help="실내로 보이는 이미지만 채택")
    parser.add_argument("--workers", type=int, default=16, help="이미지 동시 다운로드 수")
    return parser.parse_args()


def ensure_annotations(ann_dir: Path, split: str) -> Path:
    ann_file = ann_dir / "annotations" / f"instances_{split}.json"
    if ann_file.exists():
        print(f"어노테이션 재사용: {ann_file}")
        return ann_file

    ann_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ann_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print(f"어노테이션 다운로드 중 (약 241MB)... {ANN_URL}")
        with urlopen(ANN_URL) as resp, open(zip_path, "wb") as fp:
            downloaded = 0
            while chunk := resp.read(1 << 20):
                fp.write(chunk)
                downloaded += len(chunk)
                print(f"\r  {downloaded / 1e6:.0f} MB", end="")
        print()

    print("압축 해제 중...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ann_dir)
    return ann_file


def main() -> None:
    args = parse_args()
    ann_file = ensure_annotations(args.ann_dir, args.split)

    print("어노테이션 로드 중 (train2017 은 ~450MB, 1~2분 걸린다)...")
    data = json.loads(ann_file.read_text(encoding="utf-8"))

    images = {img["id"]: img for img in data["images"]}

    # image_id -> 해당 이미지의 (우리 클래스) 박스 목록, 그리고 전체 category 집합
    per_image: dict[int, list] = {}
    categories_in_image: dict[int, set] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd"):
            continue
        img_id = ann["image_id"]
        categories_in_image.setdefault(img_id, set()).add(ann["category_id"])
        insight_id = COCO_TO_INSIGHT.get(ann["category_id"])
        if insight_id is None:
            continue
        per_image.setdefault(img_id, []).append((insight_id, ann["bbox"]))

    # 클래스별 할당량을 채운다. 희귀한 순서(table -> chair -> person)로 골라야
    # person 이미지가 먼저 다 차지하는 일이 없다.
    quota = {cid: args.max_per_class for cid in INSIGHT_NAMES}
    selected: dict[int, list] = {}
    order = [1, 0, 5]  # table, chair, person

    for target in order:
        for img_id, boxes in per_image.items():
            if quota[target] <= 0:
                break
            if img_id in selected:
                continue
            if not any(cid == target for cid, _ in boxes):
                continue
            if args.indoor_only and not (categories_in_image[img_id] & INDOOR_HINTS):
                continue

            img = images[img_id]
            w, h = img["width"], img["height"]
            lines = []
            for cid, (bx, by, bw, bh) in boxes:
                if bw * bh < args.min_box_area * w * h:
                    continue
                cx = (bx + bw / 2) / w
                cy = (by + bh / 2) / h
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
            if not lines:
                continue

            selected[img_id] = lines
            # 이 이미지가 채워주는 모든 클래스의 할당량을 함께 차감한다.
            for cid in {c for c, _ in boxes}:
                quota[cid] -= 1

    print(f"선정된 이미지: {len(selected)}장")

    images_dir = args.out / "images"
    labels_dir = args.out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    def download(item) -> bool:
        img_id, lines = item
        img = images[img_id]
        name = img["file_name"]
        dst = images_dir / name
        if not dst.exists():
            try:
                with urlopen(img["coco_url"], timeout=30) as resp:
                    dst.write_bytes(resp.read())
            except Exception as exc:  # 한 장 실패로 전체를 멈추지 않는다
                print(f"  실패 {name}: {exc}")
                return False
        (labels_dir / f"{Path(name).stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return True

    print(f"이미지 다운로드 중 (workers={args.workers})...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        done = sum(pool.map(download, selected.items()))

    counts = {cid: 0 for cid in INSIGHT_NAMES}
    for lines in selected.values():
        for line in lines:
            counts[int(line.split()[0])] += 1

    print("")
    print(f"완료: {done}/{len(selected)}장 -> {args.out}")
    for cid, name in INSIGHT_NAMES.items():
        print(f"  {cid}:{name:8s} {counts[cid]} boxes")
    print("")
    print("다음 단계: dataset/ 에 합치거나 scripts/split_dataset.py 로 분할한다.")


if __name__ == "__main__":
    main()

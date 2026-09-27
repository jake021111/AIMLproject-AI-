"""프론트 연동 확인용 테스트 픽스처 생성.

raw_dataset (prepare_dataset.py 결과) 에서 이미지 몇 장을 골라 fixtures/ 로 복사하고,
각 이미지의 정답 박스를 expected_detections.json 으로 내보낸다.
포맷 정의는 docs/MODEL_INTERFACE.md §8.

값은 **사람이 라벨링한 ground truth** 이며 모델 출력이 아니다.
앱에서 letterbox 역변환까지 마친 Rect 가 bbox_pixel 과 비슷한 위치인지,
class_id 배선이 맞는지 확인하는 용도다.

이미지 선정 기준: veryHigh 클래스(계단/에스컬레이터/엘리베이터/점자블록)가 포함된 것을
우선 고르고, 박스 개수가 너무 많지 않은(육안 확인이 가능한) 이미지를 택한다.

사용법:
    python scripts/make_test_fixtures.py
    python scripts/make_test_fixtures.py --count 8 --max-boxes 6

주의: fixtures/ 는 AI Hub 원본 이미지를 포함하므로 gitignore 대상이다.
      공개 저장소에 올리지 말고 팀 드라이브 등으로 전달한다.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = ROOT / "raw_dataset" / "images"
RAW_LABELS = ROOT / "raw_dataset" / "labels"
OUT_DIR = ROOT / "fixtures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="테스트 픽스처 생성")
    parser.add_argument("--count", type=int, default=6, help="생성할 이미지 수")
    parser.add_argument(
        "--max-boxes", type=int, default=5, help="이미지당 허용 박스 수 상한 (육안 확인 편의)"
    )
    parser.add_argument("--min-boxes", type=int, default=2, help="이미지당 최소 박스 수")
    return parser.parse_args()


def load_classes() -> tuple[dict[int, str], set[int]]:
    cfg = yaml.safe_load((ROOT / "configs" / "class_labels.yaml").read_text(encoding="utf-8"))
    names = {c["id"]: c["label"] for c in cfg["classes"]}
    very_high = {c["id"] for c in cfg["classes"] if c["priority"] == "very_high"}
    return names, very_high


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        boxes.append((int(parts[0]), *(float(p) for p in parts[1:])))
    return boxes


def main() -> None:
    args = parse_args()
    if not RAW_IMAGES.is_dir():
        raise FileNotFoundError(
            f"{RAW_IMAGES} 가 없습니다. scripts/prepare_dataset.py 를 먼저 실행하세요."
        )

    names, very_high = load_classes()

    # 후보 수집: (veryHigh 포함 여부, 클래스 다양성) 순으로 정렬해 고른다.
    candidates = []
    for label_path in sorted(RAW_LABELS.glob("*.txt")):
        boxes = read_label(label_path)
        if not (args.min_boxes <= len(boxes) <= args.max_boxes):
            continue
        image_path = RAW_IMAGES / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        class_ids = {b[0] for b in boxes}
        candidates.append(
            {
                "image": image_path,
                "boxes": boxes,
                "has_very_high": bool(class_ids & very_high),
                "variety": len(class_ids),
            }
        )

    if not candidates:
        raise RuntimeError(
            f"조건({args.min_boxes}~{args.max_boxes} boxes)에 맞는 이미지가 없습니다. "
            "--max-boxes 를 늘려보세요."
        )

    candidates.sort(key=lambda c: (c["has_very_high"], c["variety"]), reverse=True)

    # 클래스 커버리지를 넓히려고, 이미 담은 클래스만 들어있는 이미지는 뒤로 미룬다.
    picked: list[dict] = []
    covered: set[int] = set()
    for cand in candidates:
        if len(picked) >= args.count:
            break
        class_ids = {b[0] for b in cand["boxes"]}
        if class_ids - covered:
            picked.append(cand)
            covered |= class_ids
    for cand in candidates:  # 모자라면 남은 것으로 채운다
        if len(picked) >= args.count:
            break
        if cand not in picked:
            picked.append(cand)

    images_out = OUT_DIR / "images"
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    images_out.mkdir(parents=True)

    manifest = {
        "note": "ground truth (사람 라벨). 모델 출력이 아니다. docs/MODEL_INTERFACE.md §8 참조.",
        "source": "AI Hub 베리어프리존(dataSetSn=186) 실내 서브셋 — 공개 배포 금지",
        "coordinate_system": "bbox_norm / bbox_pixel 모두 원본 이미지 기준 (letterbox 좌표계 아님)",
        "images": [],
    }

    for cand in picked:
        src = cand["image"]
        # 한글·공백 파일명은 Flutter asset 등록에서 문제가 되므로 ASCII 로 정규화한다.
        dst_name = f"fixture_{len(manifest['images']) + 1:02d}.jpg"
        shutil.copy2(src, images_out / dst_name)

        with Image.open(src) as im:
            width, height = im.size

        objects = []
        for class_id, cx, cy, w, h in cand["boxes"]:
            objects.append(
                {
                    "class_id": class_id,
                    "label": names.get(class_id, f"unknown_{class_id}"),
                    "bbox_norm": {
                        "cx": round(cx, 6),
                        "cy": round(cy, 6),
                        "w": round(w, 6),
                        "h": round(h, 6),
                    },
                    "bbox_pixel": {
                        "left": round((cx - w / 2) * width, 1),
                        "top": round((cy - h / 2) * height, 1),
                        "right": round((cx + w / 2) * width, 1),
                        "bottom": round((cy + h / 2) * height, 1),
                    },
                }
            )

        manifest["images"].append(
            {
                "file": f"images/{dst_name}",
                "original_file": src.name,
                "width": width,
                "height": height,
                "objects": objects,
            }
        )

    (OUT_DIR / "expected_detections.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"fixtures/ 생성 완료 - 이미지 {len(manifest['images'])}장")
    for entry in manifest["images"]:
        labels = ", ".join(sorted({o["label"] for o in entry["objects"]}))
        print(f"  {entry['file']}  {entry['width']}x{entry['height']}  [{labels}]")
    missing = sorted(set(names) - covered)
    if missing:
        print("")
        print("커버되지 않은 클래스 (데이터 없음 또는 미선정):")
        print("  " + ", ".join(f"{i}:{names[i]}" for i in missing))


if __name__ == "__main__":
    main()

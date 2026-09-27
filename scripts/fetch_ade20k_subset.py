"""ADE20K (SceneParse150) -> insight 15-class YOLO 데이터 추출.

왜 ADE20K 인가:
    AI Hub 186(실내 베리어프리)에 없는 클래스를 메운다.
      escalator  42장  <- AI Hub 에 아예 없던 클래스
      counter   367장  <- AI Hub 에서 박스 1개뿐이라 사실상 죽어 있던 클래스
      person   5075장 / table 4269장 / chair 3261장  <- 역시 AI Hub 에 없거나 부족
    매핑표는 configs/ade20k_map.yaml 에 있다.

세그멘테이션 -> bbox:
    ADE20K 는 인스턴스 구분이 없는 semantic segmentation 이다.
    annotation PNG 의 픽셀값이 곧 클래스 인덱스(1~150)이므로,
    대상 클래스 픽셀을 뽑아 **connected component 로 나눠** 각 덩어리의 bbox 를 만든다.
    맞닿은 의자 두 개가 하나로 합쳐지는 경우가 있는데, 면적/종횡비 필터로 최악만 걸러낸다.

배경 오염 주의:
    ADE20K 이미지에 우리 클래스가 찍혀 있어도 매핑하지 않으면 '배경'으로 학습된다.
    그래서 가능한 클래스를 **한꺼번에** 라벨링한다(매핑표의 core 전체).
    그래도 vending_machine / tactile_paving / elevator_door 는 ADE20K 에 대응이 없어
    오염을 완전히 피할 수는 없다. 그래서 --max-images 로 분량을 제한하는 게 기본이다.

도메인 차이:
    ADE20K 는 서구권 가정·사무실 위주고 우리는 한국 지하철·공공건물이다.
    AI Hub 데이터를 압도하지 않도록 넣는 양을 조절할 것.

사용법:
    python scripts/fetch_ade20k_subset.py --out ade20k_subset
    python scripts/fetch_ade20k_subset.py --out ade20k_subset --only escalator,counter
    python scripts/fetch_ade20k_subset.py --out ade20k_subset --max-images 1500 --include-loose
"""
import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import yaml
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
ZIP_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADE20K -> insight 15-class YOLO 변환")
    parser.add_argument("--out", type=Path, default=ROOT / "ade20k_subset")
    parser.add_argument("--src", type=Path, default=ROOT / "ade20k_raw",
                        help="ADEChallengeData2016 를 받아 풀어둘 경로 (재실행 시 재사용)")
    parser.add_argument("--splits", default="training,validation",
                        help="처리할 ADE20K split (쉼표 구분)")
    parser.add_argument("--only", default=None,
                        help="이 insight 클래스만 추출 (쉼표 구분, 예: escalator,counter). "
                             "지정해도 같은 이미지 안의 다른 core 클래스는 함께 라벨링한다.")
    parser.add_argument("--include-loose", action="store_true",
                        help="매핑표의 loose 항목도 사용 (개념 어긋남 위험, yaml 의 why 참고)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="출력 이미지 수 상한 (0=제한 없음). 희귀 클래스 우선으로 고른다.")
    parser.add_argument("--min-box-area", type=float, default=0.002,
                        help="이미지 면적 대비 최소 박스 비율")
    parser.add_argument("--max-aspect", type=float, default=15.0,
                        help="종횡비 상한 (비정상적으로 가늘고 긴 덩어리 제거)")
    return parser.parse_args()


def ensure_dataset(src: Path) -> Path:
    root = src / "ADEChallengeData2016"
    if (root / "images").is_dir() and (root / "annotations").is_dir():
        print(f"ADE20K 재사용: {root}")
        return root

    src.mkdir(parents=True, exist_ok=True)
    zip_path = src / "ADEChallengeData2016.zip"
    if not zip_path.exists():
        print(f"ADE20K 다운로드 중 (약 923MB)... {ZIP_URL}")
        with urlopen(ZIP_URL) as resp, open(zip_path, "wb") as fp:
            done = 0
            while chunk := resp.read(1 << 20):
                fp.write(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e6:.0f} MB", end="")
        print()

    print("압축 해제 중...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(src)
    return root


def load_mapping(include_loose: bool) -> dict[int, tuple[int, str]]:
    """ade_idx -> (insight_id, insight_name)"""
    cfg = yaml.safe_load((ROOT / "configs" / "ade20k_map.yaml").read_text(encoding="utf-8"))
    entries = list(cfg["core"])
    if include_loose:
        entries += list(cfg["loose"])
    return {e["ade_idx"]: (e["insight_id"], e["insight"]) for e in entries}


def connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """이진 마스크에서 연결 요소별 bbox(x1,y1,x2,y2) 목록을 만든다.

    scipy 가 있으면 그걸 쓰고, 없으면 OpenCV 로 넘어간다. 둘 다 없으면
    전체를 한 덩어리로 취급한다(인스턴스가 붙지만 아예 못 쓰는 것보다 낫다).
    """
    try:
        from scipy import ndimage

        labeled, n = ndimage.label(mask)
        return [
            (int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop))
            for sl in ndimage.find_objects(labeled)
            if sl is not None
        ]
    except ImportError:
        pass

    try:
        import cv2

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        out = []
        for i in range(1, n):
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            out.append((int(x), int(y), int(x + w), int(y + h)))
        return out
    except ImportError:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        return [(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)]


def main() -> None:
    args = parse_args()
    root = ensure_dataset(args.src)
    mapping = load_mapping(args.include_loose)

    wanted = None
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        known = {name for _, name in mapping.values()}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"매핑에 없는 클래스: {', '.join(sorted(unknown))}\n"
                             f"사용 가능: {', '.join(sorted(known))}")

    # 1단계: 각 이미지에서 뽑히는 박스를 모은다 (파일은 아직 쓰지 않는다).
    collected: list[tuple[Path, Path, list[str], set[str]]] = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ann_dir = root / "annotations" / split
        img_dir = root / "images" / split
        if not ann_dir.is_dir():
            print(f"건너뜀 (없음): {ann_dir}")
            continue

        anns = sorted(ann_dir.glob("*.png"))
        print(f"{split}: {len(anns)}장 스캔 중...")
        for i, ann_path in enumerate(anns, 1):
            if i % 2000 == 0:
                print(f"  {i}/{len(anns)}")

            ann = np.array(Image.open(ann_path))
            present = set(np.unique(ann)) & set(mapping)
            if not present:
                continue
            names_here = {mapping[idx][1] for idx in present}
            if wanted and not (names_here & wanted):
                continue

            img_path = img_dir / f"{ann_path.stem}.jpg"
            if not img_path.exists():
                continue

            h, w = ann.shape[:2]
            lines = []
            for ade_idx in sorted(present):
                insight_id, _ = mapping[ade_idx]
                for x1, y1, x2, y2 in connected_components(ann == ade_idx):
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 1 or bh <= 1:
                        continue
                    if bw * bh < args.min_box_area * w * h:
                        continue
                    if max(bw / bh, bh / bw) > args.max_aspect:
                        continue
                    lines.append(
                        f"{insight_id} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} "
                        f"{bw / w:.6f} {bh / h:.6f}"
                    )
            if lines:
                got = {mapping[i2][1] for i2 in present}
                collected.append((img_path, ann_path, lines, got))

    print(f"\n박스가 나온 이미지: {len(collected)}장")

    # 2단계: 상한이 있으면 희귀 클래스가 든 이미지를 먼저 남긴다.
    if args.max_images and len(collected) > args.max_images:
        freq = Counter()
        for _, _, _, names in collected:
            freq.update(names)
        # 가장 희귀한 클래스의 빈도를 점수로 써서 오름차순 정렬(희귀할수록 앞)
        collected.sort(key=lambda c: min(freq[n] for n in c[3]))
        collected = collected[: args.max_images]
        print(f"상한 적용 -> {len(collected)}장 (희귀 클래스 우선)")

    # 3단계: 기록
    images_out = args.out / "images"
    labels_out = args.out / "labels"
    if args.out.exists():
        shutil.rmtree(args.out)
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    box_count = Counter()
    for img_path, _, lines, _ in collected:
        name = f"ade_{img_path.stem}"
        shutil.copy2(img_path, images_out / f"{name}.jpg")
        (labels_out / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for line in lines:
            box_count[int(line.split()[0])] += 1

    names = {v[0]: v[1] for v in mapping.values()}
    print(f"\n완료: {len(collected)}장 -> {args.out}")
    print("클래스별 박스 수")
    for cid in sorted(box_count):
        print(f"  {cid:2d} {names.get(cid, '?'):16s} {box_count[cid]}")
    print("\n다음: dataset/ 에 합치거나 scripts/split_dataset.py 로 분할한다.")
    print("주의: ADE20K 는 서구권 실내 위주다. AI Hub 분량을 압도하지 않게 --max-images 로 조절할 것.")


if __name__ == "__main__":
    main()

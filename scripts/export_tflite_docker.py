"""Linux 컨테이너에서 Ultralytics 공식 TFLite INT8 변환을 수행한다 (CLAUDE.md §7.4).

왜 Docker 인가:
    Ultralytics 의 TFLite/LiteRT export 는 Linux x86 / macOS 에서만 동작한다
    (exporter.py: `assert MACOS or (LINUX and not ARM64)`). 개발 PC가 Windows 라서
    로컬에서는 `yolo export format=tflite int8=True` 한 줄을 쓸 수 없다.

    Windows 에서 ONNX -> onnx2tf -> TFLiteConverter 로 우회하는 경로도 만들어 두었지만
    (scripts/onnx_to_tflite.py), 그 경로의 INT8 산출물은 출력이 무너진다.
    INT8 이 요구사항이므로 공식 파이프라인을 컨테이너 안에서 그대로 돌린다.

전제:
    Docker Desktop 이 실행 중이어야 한다 (`docker info` 가 성공해야 함).

한글 경로 주의:
    프로젝트 경로에 한글이 있으면(…/졸푸바오/ai) 볼륨 마운트와 컨테이너 내부 툴체인
    양쪽에서 문제가 생긴다. 그래서 필요한 파일만 ASCII 임시 폴더로 복사해 마운트하고
    결과만 회수한다.

사용법:
    python scripts/export_tflite_docker.py --weights runs/indoor15/weights/best.pt
    python scripts/export_tflite_docker.py --weights runs/indoor15/weights/best.pt --samples 200
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE = "ultralytics/ultralytics:latest-cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Docker 안에서 공식 TFLite INT8 변환")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/indoor15/weights/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--samples", type=int, default=300,
                        help="INT8 대표 데이터셋으로 컨테이너에 넣을 val 이미지 수")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="사용할 Docker 이미지")
    parser.add_argument("--out", type=Path, default=ROOT / "export_out")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="임시 작업 폴더를 지우지 않는다 (디버깅용)")
    return parser.parse_args()


def require_docker() -> None:
    r = subprocess.run(["docker", "info", "--format", "{{.OSType}}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            "Docker 데몬에 연결할 수 없습니다. Docker Desktop 을 실행한 뒤 다시 시도하세요.\n"
            f"  {(r.stderr or '').strip()[:300]}"
        )
    if "linux" not in (r.stdout or "").lower():
        raise SystemExit(
            f"Linux 컨테이너가 필요합니다 (현재: {r.stdout.strip()}).\n"
            "Docker Desktop 에서 'Switch to Linux containers' 를 선택하세요."
        )


def stage_calibration(args, work: Path) -> None:
    """대표 데이터셋 이미지와 라벨을 ASCII 임시 폴더로 복사하고 data.yaml 을 쓴다."""
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    base = (args.data.parent / cfg["path"]).resolve()
    val_images = base / cfg["val"]
    val_labels = Path(str(val_images).replace("images", "labels"))

    dst_images = work / "dataset" / "images" / "val"
    dst_labels = work / "dataset" / "labels" / "val"
    dst_images.mkdir(parents=True)
    dst_labels.mkdir(parents=True)

    images = sorted(p for p in val_images.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"val 이미지가 없습니다: {val_images}")
    step = max(1, len(images) // args.samples)
    picked = images[::step][: args.samples]

    for i, src in enumerate(picked):
        # 한글·공백 파일명은 컨테이너 안에서 말썽을 일으킬 수 있으니 ASCII 로 바꾼다.
        name = f"calib_{i:04d}"
        shutil.copy2(src, dst_images / f"{name}.jpg")
        lbl = val_labels / f"{src.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, dst_labels / f"{name}.txt")

    # train 경로도 있어야 ultralytics 가 yaml 검증을 통과한다. val 을 그대로 가리킨다.
    (work / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": "/work/dataset",
                "train": "images/val",
                "val": "images/val",
                "names": cfg["names"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print(f"대표 데이터셋: {len(picked)}장 준비")


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"가중치 파일이 없습니다: {args.weights}")
    require_docker()

    work = Path(tempfile.mkdtemp(prefix="yoloexport_"))
    if not str(work).isascii():
        raise SystemExit(f"임시 경로에 비ASCII 문자가 있습니다: {work}")

    try:
        shutil.copy2(args.weights, work / "best.pt")
        stage_calibration(args, work)

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{work}:/work",
            "-w", "/work",
            args.image,
            "yolo", "export",
            "model=/work/best.pt",
            "format=tflite",
            "int8=True",
            f"imgsz={args.imgsz}",
            "nms=False",
            "data=/work/data.yaml",
        ]
        print("실행:", " ".join(cmd))
        print("(첫 실행은 이미지 다운로드로 수 분 걸린다)")
        if subprocess.run(cmd).returncode != 0:
            raise SystemExit("docker 안에서 yolo export 실패")

        produced = sorted(work.rglob("*_int8.tflite")) or sorted(work.rglob("*.tflite"))
        if not produced:
            raise SystemExit(f"tflite 산출물을 찾지 못했습니다: {work}")

        args.out.mkdir(parents=True, exist_ok=True)
        print("")
        print("=" * 64)
        for f in produced:
            dst = args.out / f.name
            shutil.copy2(f, dst)
            print(f"  {dst}  ({f.stat().st_size / 1e6:.1f} MB)")
        print("=" * 64)
        print("  검증:")
        print(f"    python scripts/verify_tflite.py --model {args.out}/<파일> \\")
        print("        --image fixtures/images/fixture_01.jpg")
    finally:
        if args.keep_workdir:
            print(f"작업 폴더 유지: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

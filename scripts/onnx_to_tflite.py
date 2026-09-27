"""ONNX -> TFLite(INT8) 변환 — Windows 우회 경로.

왜 이 스크립트가 필요한가:
    Ultralytics 의 `model.export(format="tflite"/"litert")` 는 **Linux x86 / macOS 에서만** 동작한다
    (exporter.py: `assert MACOS or (LINUX and not ARM64)`). 개발 PC가 Windows 라서
    export.py 가 그대로는 실패한다. ONNX export 는 Windows 에서 되므로 그 뒤를 직접 잇는다.

파이프라인:
    best.pt --(yolo_onnx.py: bbox 정규화 포함)--> best.onnx
            --(onnx2tf)--> TF SavedModel --(TFLiteConverter + 대표 데이터셋)--> best_int8.tflite

실측으로 확인한 사항들:

  1) bbox 는 0~1 로 정규화해서 내보낸다
     Ultralytics raw 출력은 bbox 가 letterbox 640 픽셀(0~640), 클래스 점수는 0~1 인데
     둘이 `[1, 4+nc, 8400]` 한 텐서에 섞여 있다. TFLite INT8 은 per-tensor 양자화라
     scale 이 640 쪽에 맞춰지고(실측 2.505) 클래스 점수가 뭉개진다.
     -> scripts/yolo_onnx.py 가 ONNX 내보내기 시점에 bbox 를 imgsz 로 나눈다.
        docs/MODEL_INTERFACE.md §3.1 의 "bbox 0~1" 계약이 이걸로 유지된다.
        (FP32/FP16 에서 bbox 범위 0.0042~1.0014 로 정상 동작 확인.)

  2) onnx2tf 의 -oiqt 직접 양자화는 쓰지 않는다
     onnx2tf 가 직접 뽑은 INT8 tflite 는 sigmoid(LOGISTIC) 노드에서 로드가 거부된다:
       "output->params.scale == 1. / 256 was not true. Node ... (LOGISTIC) failed to prepare"
     -> onnx2tf 는 SavedModel 까지만 만들고 양자화는 TensorFlow 자체 컨버터로 한다.
        (이러면 로드는 되지만, 아래 3번 문제가 남는다.)

  3) ⚠️ 이 경로의 INT8 결과물은 현재 **쓸 수 없다** — 기본값이 float16 인 이유
     TF 컨버터로 만든 INT8 모델은 로드·실행은 되지만 출력이 무너진다:
       bbox 4행이 전부 0, 클래스 점수 최대 0.30 -> 0.16
     같은 ONNX 로 만든 FP32/FP16 은 정상이므로(bbox 0.0042~1.0014, 점수 0.3040)
     INT8 양자화 단계 자체의 문제다. YOLOv8 의 bbox 브랜치(DFL: softmax + matmul)가
     per-tensor INT8 양자화에 취약한 것으로 보인다.
     정규화를 TF 래퍼에서 하든 ONNX 단계에서 하든, 입출력 타입을 int8 로 두든
     float32 로 두든 결과가 동일했다(전부 실측 확인).
     -> 당분간 **float16 을 쓴다** (6.4MB, GPU delegate, 정확도 손실 사실상 없음).
        INT8(3.4MB, NNAPI)이 필요하면 Linux/Colab 에서 Ultralytics 공식
        `yolo export format=tflite int8=True` 로 뽑아 비교할 것.

INT8 대표 데이터셋(calibration):
    data.yaml 의 val 이미지에서 N장을 뽑아 letterbox + RGB + 0~1 정규화한다.
    앱의 전처리(docs/MODEL_INTERFACE.md §2)와 **동일한 방식**이어야 양자화 오차가 줄어든다.

사용법:
    python export.py --weights runs/indoor15/weights/best.pt      # ONNX 생성
    python scripts/onnx_to_tflite.py --onnx runs/indoor15/weights/best.onnx

    python scripts/onnx_to_tflite.py --onnx yolov8n.onnx --quant none   # FP32 (디버깅용)
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX -> TFLite INT8 (Windows 우회)")
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    parser.add_argument("--samples", type=int, default=64, help="캘리브레이션 이미지 수")
    parser.add_argument(
        "--quant",
        default="int8_hybrid",
        choices=["int8_hybrid", "int8_full", "float16", "none"],
        help="int8_hybrid(기본): 가중치 INT8 / 활성 float — 3.4MB, 동작 확인됨. "
        "CLAUDE.md §7.4 가 규정한 Hybrid Quantization 경로다. "
        "int8_full: 가중치+활성 모두 INT8 (NNAPI 대상). 이 Windows 경로에서는 출력이 깨진다. "
        "float16: 6.4MB, GPU delegate. none: FP32 (12.8MB, 디버깅용)",
    )
    parser.add_argument("--no-calib", action="store_true",
                        help="(deprecated) --quant none 과 같다")
    parser.add_argument(
        "--io-type",
        default="float32",
        choices=["float32", "int8"],
        help="--quant int8 일 때 입출력 텐서 타입. 가중치·연산은 어느 쪽이든 INT8 이다.",
    )
    parser.add_argument("--calib-dir", type=Path, default=None,
                        help="캘리브레이션 이미지 폴더 (기본: data.yaml 의 val 경로)")
    parser.add_argument("--out", type=Path, default=ROOT / "stub_models")
    return parser.parse_args()


def is_ascii_path(path: Path) -> bool:
    return str(path).isascii()


def letterbox(img: Image.Image, size: int) -> np.ndarray:
    """앱과 동일한 letterbox: 비율 유지 + 회색(114) 패딩. RGB, 0~1 float32 반환."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR),
                 ((size - nw) // 2, (size - nh) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def collect_calibration_images(args) -> list[Path]:
    if args.calib_dir is not None:
        val_dir = args.calib_dir
    else:
        cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
        val_dir = (args.data.parent / cfg["path"] / cfg["val"]).resolve()

    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"캘리브레이션 이미지 경로가 없습니다: {val_dir}\n"
            "scripts/split_dataset.py 를 먼저 실행하거나 --calib-dir / --no-calib 을 쓰세요."
        )

    images = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"{val_dir} 에 이미지가 없습니다.")

    # 고르게 뽑는다 (앞쪽 N장만 쓰면 촬영 시점이 몰려 대표성이 떨어진다).
    step = max(1, len(images) // args.samples)
    return images[::step][: args.samples]


def onnx_to_savedmodel(onnx_path: Path, work: Path) -> Path:
    """onnx2tf 로 SavedModel 만 만든다 (양자화는 하지 않는다)."""
    work_onnx = work / onnx_path.name
    shutil.copy2(onnx_path, work_onnx)
    saved = work / "saved_model"

    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(work_onnx),
        "-o", str(saved),
        "-nuo",     # onnx shape inference 생략 (yolo 에서 간혹 실패)
        "-fdosm",   # 최신 onnx2tf 는 기본(flatbuffer_direct)에서 SavedModel 을 안 만든다
        "-osd",     # serving_default 시그니처 부여 (TFLiteConverter.from_saved_model 에 필요)
    ]
    print("실행:", " ".join(cmd))
    if subprocess.run(cmd, cwd=work).returncode != 0:
        raise SystemExit("onnx2tf (SavedModel 변환) 실패")
    return saved


def main() -> None:
    args = parse_args()
    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX 파일이 없습니다: {args.onnx}")

    if args.no_calib:
        args.quant = "none"
    args.out.mkdir(parents=True, exist_ok=True)
    calib_images = collect_calibration_images(args) if args.quant == "int8_full" else None

    import tensorflow as tf

    # ai_edge_litert / TF 의 C++ 레이어는 **비ASCII 경로의 파일을 열지 못한다.**
    # 이 프로젝트 경로에는 한글이 들어 있어서(…/졸푸바오/ai) 그대로 두면
    # "Could not open '…\\??Ǫ?ٿ?\\…'" 로 죽는다. ASCII 임시 디렉터리에서 작업한다.
    with tempfile.TemporaryDirectory(prefix="onnx2tf_") as tmp:
        work = Path(tmp)
        if not is_ascii_path(work):
            raise SystemExit(
                f"임시 경로에도 비ASCII 문자가 있습니다: {work}\n"
                "TMP 환경변수를 ASCII 경로로 바꾼 뒤 다시 실행하세요."
            )

        saved = onnx_to_savedmodel(args.onnx, work)
        imgsz = args.imgsz

        # SavedModel 을 그대로 변환한다.
        # bbox 정규화는 ONNX 그래프 안에서 이미 끝나 있다 (scripts/yolo_onnx.py).
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))

        if args.quant == "none":
            suffix = "float32"
        elif args.quant == "int8_hybrid":
            # Dynamic Range Quantization: 가중치만 INT8 로 저장하고 활성은 float 로 둔다.
            # 대표 데이터셋이 필요 없고, full integer 와 **크기가 같다**(둘 다 3.4MB).
            # full integer 가 이 경로에서 깨지는 반면 이건 정상 동작한다 (실측).
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            suffix = "int8_hybrid"
        elif args.quant == "float16":
            # 대표 데이터셋이 필요 없다. 가중치만 FP16 으로 줄이고 연산은 FP32 로 돈다.
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            suffix = "float16"
        else:
            arr = np.stack([letterbox(Image.open(p), imgsz) for p in calib_images])
            print(f"캘리브레이션: {arr.shape} ({arr.nbytes / 1e6:.0f} MB, {len(calib_images)}장)")

            def representative_dataset():
                for sample in arr:
                    yield [sample[None, ...]]

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            io_dtype = tf.int8 if args.io_type == "int8" else tf.float32
            converter.inference_input_type = io_dtype
            converter.inference_output_type = io_dtype
            suffix = "int8" if args.io_type == "int8" else "int8_iofp32"

        print("TFLite 변환 중...")
        tflite_model = converter.convert()

    dst = args.out / f"{args.onnx.stem}_{suffix}.tflite"
    dst.write_bytes(tflite_model)

    print("")
    print("=" * 64)
    print(f"  생성: {dst}  ({len(tflite_model) / 1e6:.1f} MB)")
    print("=" * 64)
    print("  검증:")
    print(f"    python scripts/verify_tflite.py --model {dst} \\")
    print("        --image fixtures/images/fixture_01.jpg")
    print("  스펙: docs/MODEL_INTERFACE.md")


if __name__ == "__main__":
    main()

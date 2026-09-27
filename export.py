"""best.pt -> TFLite 변환 스크립트 (CLAUDE.md §7.4).

플랫폼별 경로가 다르다:

  Linux x86 / macOS
      `yolo export ... format=tflite` 한 줄이 PyTorch -> ONNX -> TF SavedModel -> TFLite(+PTQ)
      를 전부 수행한다. 이 스크립트를 그냥 실행하면 된다.

  Windows  ← 현재 개발 PC
      Ultralytics 가 LiteRT/TFLite export 를 막아 둔다
      (exporter.py: `assert MACOS or (LINUX and not ARM64)`).
      ONNX 까지만 만든 뒤 scripts/onnx_to_tflite.py 로 이어서 변환한다.
      이 스크립트는 Windows 를 감지하면 자동으로 --onnx-only 모드로 동작한다.

NMS 는 항상 모델 밖(Dart)에서 처리하므로 export 에 포함하지 않는다 (INT8 양자화 호환성 문제).

기본은 INT8 (NPU/DSP·NNAPI 가속, ~3MB).
INT8 변환 후 validate.py 결과 mAP 낙폭이 5%를 초과하면 --format fp16 으로 재시도
(GPU delegate 경로, hybrid 개념의 fallback).

사용법:
    python export.py --weights runs/indoor15/weights/best.pt
    python export.py --weights runs/indoor15/weights/best.pt --format fp16
    python export.py --weights runs/indoor15/weights/best.pt --onnx-only
"""
import argparse
import platform
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8n -> TFLite 변환")
    parser.add_argument("--weights", default="runs/indoor15/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--data", default="data.yaml", help="INT8 대표 데이터셋 샘플링용 (val 경로 사용)")
    parser.add_argument(
        "--format",
        default="int8",
        choices=["int8", "fp16"],
        help="int8: Full Integer Quantization, NPU/DSP(NNAPI) 가속용 (기본) / "
        "fp16: GPU delegate 가속용, INT8 정확도 손실이 클 때 fallback",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="ONNX 까지만 변환한다 (이후 scripts/onnx_to_tflite.py 로 이어감). "
        "Windows 에서는 지정하지 않아도 자동 적용된다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weights}")

    model = YOLO(str(weights))

    # Ultralytics 의 TFLite/LiteRT export 는 Linux x86 / macOS 전용이다.
    onnx_only = args.onnx_only or platform.system() == "Windows"

    if onnx_only:
        # ⚠️ ultralytics 기본 ONNX export 를 쓰면 안 된다. bbox 가 letterbox 640 픽셀 단위로
        #    나와서 앱 계약(docs/MODEL_INTERFACE.md §3.1: bbox 0~1)과 어긋나고,
        #    INT8 양자화 시 클래스 점수가 뭉개진다. 정규화 포함 export 를 쓴다.
        sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
        from yolo_onnx import export_normalized_onnx

        exported = export_normalized_onnx(
            model, weights.with_suffix(".onnx"), imgsz=args.imgsz, opset=13
        )
        print("")
        print("=" * 64)
        print(f"  ONNX 생성 완료 (bbox 0~1 정규화 포함): {exported}")
        if platform.system() == "Windows":
            print("  (Windows 는 Ultralytics TFLite export 미지원 -> ONNX 경로로 우회)")
        print("")
        print("  다음 단계:")
        print(f"    python scripts/onnx_to_tflite.py --onnx {exported}")
        print("=" * 64)
        return

    if args.format == "int8":
        # data.yaml val 경로에서 대표 데이터셋 300장을 자동 샘플링한다.
        model.export(format="tflite", int8=True, imgsz=args.imgsz, nms=False, data=args.data)
    else:
        model.export(format="tflite", half=True, imgsz=args.imgsz, nms=False)


if __name__ == "__main__":
    main()

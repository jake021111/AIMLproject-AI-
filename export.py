"""best.pt -> TFLite 변환 스크립트 (CLAUDE.md §7.4).

`yolo export ... format=tflite` 한 줄이 내부적으로
PyTorch -> ONNX -> TF SavedModel -> TFLite (+PTQ) 를 전부 수행한다.
NMS는 항상 모델 밖(Dart)에서 처리하므로 export에 포함하지 않는다 (INT8 양자화 호환성 문제).

기본은 INT8 (NPU/DSP·NNAPI 가속, ~3MB).
INT8 변환 후 validate.py 결과 mAP 낙폭이 5%를 초과하면 --format fp16 으로 재시도
(GPU delegate 경로, hybrid 개념의 fallback).

사용법:
    python export.py --weights runs/indoor15/weights/best.pt
    python export.py --weights runs/indoor15/weights/best.pt --format fp16
"""
import argparse
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weights}")

    model = YOLO(str(weights))

    if args.format == "int8":
        # data.yaml val 경로에서 대표 데이터셋 300장을 자동 샘플링한다.
        model.export(format="tflite", int8=True, imgsz=args.imgsz, nms=False, data=args.data)
    else:
        model.export(format="tflite", half=True, imgsz=args.imgsz, nms=False)


if __name__ == "__main__":
    main()

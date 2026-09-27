"""프론트 연동 검증용 스텁 TFLite 모델 생성 (학습 불필요).

실제 학습이 끝나기 전에 Flutter 쪽에서 전처리/추론/디코드/NMS 배선을 완성할 수 있도록,
**최종 모델과 입출력 구조가 동일한** tflite 파일을 만든다. 상세는 docs/MODEL_INTERFACE.md §6.

두 가지 모드:

  stub15 (기본)
      configs/class_labels.yaml 의 15 클래스로 YOLOv8n 헤드를 구성한 랜덤 가중치 모델.
      출력 shape [1, 19, 8400] — 최종 모델과 완전히 동일하므로 파일 교체만으로 전환된다.
      학습을 안 했으니 탐지 결과 자체는 의미가 없다. shape/라벨 배선 검증용.

  coco80
      COCO 2017 사전학습 yolov8n 그대로 변환. 출력 shape [1, 84, 8400].
      실제 가중치라 박스가 "맞게" 나오므로 letterbox 역변환·NMS 로직의 정확성을
      눈으로 검증할 수 있다. 앱에서 numClasses 를 출력 shape(= shape[1] - 4)로
      유도해 두면 이 모델로도 그대로 돌아간다.

사용법:
    python scripts/export_stub_model.py --mode stub15
    python scripts/export_stub_model.py --mode coco80
    python scripts/export_stub_model.py --mode stub15 --no-int8   # FP32 (양자화 파라미터 없음)

주의:
  - TFLite export 는 tensorflow / onnx2tf 등에 의존한다. Python 3.10~3.12 환경에서 실행할 것
    (3.13+ 에서는 tensorflow 휠이 없어 실패한다). Colab 기본 런타임이면 그대로 동작한다.
  - INT8 양자화는 대표 데이터셋(calibration)이 필요하다.
      stub15 -> data.yaml 의 val 경로 (scripts/split_dataset.py 를 먼저 실행해야 한다)
      coco80 -> coco8.yaml (ultralytics 가 8장짜리 샘플을 자동 다운로드, 네트워크 필요)
    준비가 안 됐으면 --no-int8 로 FP32 스텁을 먼저 넘겨도 배선 검증은 가능하다
    (단, 앱의 dequantize 분기는 실제 INT8 모델로 다시 확인해야 한다).
"""
import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "stub_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="연동 검증용 스텁 TFLite 생성")
    parser.add_argument("--mode", default="stub15", choices=["stub15", "coco80"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--data",
        default=None,
        help="INT8 대표 데이터셋 yaml (기본: stub15=data.yaml, coco80=coco8.yaml)",
    )
    parser.add_argument(
        "--quant",
        default="float16",
        choices=["float16", "int8", "none"],
        help="scripts/onnx_to_tflite.py 와 동일. 기본 float16 "
        "(Windows 경로의 INT8 은 현재 출력이 깨진다 — README '변환 시 주의' 4번)",
    )
    return parser.parse_args()


def build_model(mode: str):
    """mode 에 맞는 YOLO 모델 객체를 만든다."""
    from ultralytics import YOLO

    if mode == "coco80":
        # COCO 2017 사전학습 가중치 그대로 (80 classes).
        return YOLO("yolov8n.pt"), 80

    # stub15: 15-class 헤드로 아키텍처만 구성한다 (랜덤 초기화).
    # ultralytics 패키지 안의 yolov8.yaml 을 통째로 복사해 nc 만 덮어쓴다.
    # (nc 한 줄짜리 yaml 로는 backbone/head 가 정의되지 않아 parse_model 이 실패한다.)
    import ultralytics

    class_yaml = ROOT / "configs" / "class_labels.yaml"
    classes = yaml.safe_load(class_yaml.read_text(encoding="utf-8"))["classes"]
    nc = len(classes)

    pkg_yaml = Path(ultralytics.__file__).parent / "cfg" / "models" / "v8" / "yolov8.yaml"
    if not pkg_yaml.exists():
        raise FileNotFoundError(f"ultralytics 모델 yaml 을 찾을 수 없습니다: {pkg_yaml}")

    spec = yaml.safe_load(pkg_yaml.read_text(encoding="utf-8"))
    spec["nc"] = nc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 파일명에 'yolov8n' 이 들어가야 ultralytics 가 n 스케일로 해석한다.
    stub_yaml = OUT_DIR / f"yolov8n_stub{nc}.yaml"
    stub_yaml.write_text(
        f"# 자동 생성 — {pkg_yaml.name} 복사 + nc={nc} ({class_yaml.name} 기준)\n"
        + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # 가중치를 로드하지 않으므로 랜덤 초기화 상태다.
    return YOLO(str(stub_yaml)), nc


def main() -> None:
    args = parse_args()
    model, nc = build_model(args.mode)
    expected_channels = 4 + nc

    data = args.data
    if data is None:
        data = "coco8.yaml" if args.mode == "coco80" else str(ROOT / "data.yaml")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"{args.mode}_stub_{args.quant}.tflite"

    # Ultralytics 의 TFLite/LiteRT export 는 Linux x86 / macOS 전용이다.
    # Windows 에서는 ONNX 까지만 만들고 onnx2tf 로 이어서 변환한다 (CLAUDE.md §14).
    if platform.system() == "Windows":
        # bbox 정규화 포함 ONNX (scripts/yolo_onnx.py). ultralytics 기본 export 는
        # bbox 를 픽셀 단위로 내보내 앱 계약(0~1)과 어긋난다.
        sys.path.insert(0, str(ROOT / "scripts"))
        from yolo_onnx import export_normalized_onnx

        onnx_path = export_normalized_onnx(
            model, OUT_DIR / f"{args.mode}_norm.onnx", imgsz=args.imgsz
        )
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "onnx_to_tflite.py"),
            "--onnx", str(onnx_path),
            "--imgsz", str(args.imgsz),
            "--out", str(OUT_DIR),
            "--quant", args.quant,
        ]
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            raise SystemExit("onnx_to_tflite.py 실패")

        produced = OUT_DIR / f"{onnx_path.stem}_{args.quant}.tflite"
        if not produced.exists():
            raise SystemExit(f"tflite 산출물을 찾지 못했습니다: {produced}")
        shutil.move(str(produced), dst)
    else:
        if args.quant == "int8":
            exported = model.export(
                format="tflite", int8=True, imgsz=args.imgsz, nms=False, data=data
            )
        elif args.quant == "float16":
            exported = model.export(format="tflite", half=True, imgsz=args.imgsz, nms=False)
        else:
            exported = model.export(format="tflite", imgsz=args.imgsz, nms=False)
        shutil.copy2(Path(exported), dst)

    size_mb = dst.stat().st_size / (1024 * 1024)
    print("")
    print("=" * 62)
    print(f"  스텁 모델 생성 완료: {dst.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    print(f"  모드      : {args.mode} (nc={nc})")
    print(f"  예상 출력 : [1, {expected_channels}, 8400]   # {expected_channels} = 4 + {nc}")
    print(f"  예상 입력 : [1, {args.imgsz}, {args.imgsz}, 3]  NHWC / RGB")
    print("=" * 62)
    if args.mode == "stub15":
        print("  ! 랜덤 가중치다. 탐지 결과는 의미 없고 shape/배선 검증 전용이다.")
    else:
        print("  ! 80-class COCO 모델이다. 라벨은 COCO 기준(person=0, chair=56).")
        print("    앱에서 numClasses 를 출력 shape[1]-4 로 유도해야 그대로 돌아간다.")
    print("  상세 스펙: docs/MODEL_INTERFACE.md")


if __name__ == "__main__":
    main()

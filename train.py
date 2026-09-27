"""YOLOv8n indoor-15 학습 스크립트 (CLAUDE.md §7.3).

⚠️ per-class weight 인자는 없다 (Ultralytics 미지원).
   veryHigh 클래스 recall 확보는 scripts/oversample_critical.py 로 데이터 단계에서 처리하고,
   여기서는 학습만 담당한다.
⚠️ freeze는 '레이어 개수'다. 학습 전체에 걸쳐 고정되며, 지정 epoch 이후 자동 unfreeze는 없다.

사용법:
    python train.py
    python train.py --epochs 100 --batch 16   # VRAM 8GB 이상일 때
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="insight YOLOv8n indoor-15 학습")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--weights", default="yolov8n.pt", help="사전학습 가중치 (COCO 80-class)")
    parser.add_argument("--imgsz", type=int, default=640)
    # 기본값 8: 개발 PC가 RTX 3050 Laptop(VRAM 4GB)이라 640px 학습에서 batch=32는 OOM이 난다.
    # VRAM 8GB 이상(또는 Colab T4)이면 --batch 16~32 로 올린다. (CLAUDE.md §7.3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--freeze", type=int, default=10, help="앞 N개 레이어 고정 (epoch 수 아님)")
    # 절대 경로로 준다. 상대 경로면 ultralytics 가 settings 의 runs_dir 밑에 또 중첩시켜
    # runs/detect/runs/<name> 처럼 들어가서 README·validate.py 의 경로와 어긋난다.
    parser.add_argument("--project", default=str(ROOT / "runs"))
    parser.add_argument("--name", default="indoor15")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        optimizer="SGD",
        momentum=0.937,
        lr0=0.01,
        lrf=0.01,  # 비율. 최종 LR = lr0 * lrf = 0.0001
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=10,  # 마지막 10 epoch mosaic 비활성화 (수렴 안정)
        freeze=args.freeze,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

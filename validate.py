"""mAP 검증 + veryHigh 클래스 recall 체크 (CLAUDE.md §7.5).

목표: mAP@0.5 >= 0.70 (실내 테스트 기준)
      veryHigh(stairs/escalator/elevator_door/tactile_paving) recall >= 0.85

best.pt 와 INT8 변환본(best_saved_model/best_int8.tflite) 양쪽에 동일하게 돌려서
INT8 변환으로 인한 mAP 낙폭(-3~5% 허용)을 비교 확인할 것.

사용법:
    python validate.py --weights runs/indoor15/weights/best.pt
    python validate.py --weights runs/indoor15/weights/best_saved_model/best_int8.tflite
"""
import argparse

from ultralytics import YOLO

VERY_HIGH_CLASSES = {"stairs", "escalator", "elevator_door", "tactile_paving"}
TARGET_MAP50 = 0.70
TARGET_VERY_HIGH_RECALL = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mAP / 클래스별 recall 검증")
    parser.add_argument("--weights", required=True, help="best.pt 또는 best_int8.tflite")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split)

    names = metrics.names
    recalls = metrics.box.r
    map50 = metrics.box.map50

    print(f"\n=== mAP@0.5 = {map50:.3f} (목표 >= {TARGET_MAP50}) ===")
    print("클래스별 Recall:")

    very_high_recalls = []
    for idx, recall in enumerate(recalls):
        label = names[idx]
        flag = "  *** veryHigh ***" if label in VERY_HIGH_CLASSES else ""
        print(f"  {label:16s}: recall={recall:.3f}{flag}")
        if label in VERY_HIGH_CLASSES:
            very_high_recalls.append(recall)

    if very_high_recalls:
        avg_very_high = sum(very_high_recalls) / len(very_high_recalls)
        status = "OK" if avg_very_high >= TARGET_VERY_HIGH_RECALL else "FAIL"
        print(f"\nveryHigh 평균 recall = {avg_very_high:.3f} (목표 >= {TARGET_VERY_HIGH_RECALL}) [{status}]")


if __name__ == "__main__":
    main()

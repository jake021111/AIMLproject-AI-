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

    # ⚠️ metrics.box.r 은 **평가 데이터에 실제로 등장한 클래스만** 담는 축약 배열이다.
    #    enumerate 인덱스를 class id 로 쓰면 라벨이 밀린다.
    #    (현재 데이터셋은 15개 중 table/person/escalator/automatic_door/kiosk 가 비어 있어
    #     이 버그가 그대로 드러난다.) ap_class_index 로 실제 class id 를 얻어야 한다.
    class_ids = list(metrics.box.ap_class_index)

    print(f"\n=== mAP@0.5 = {map50:.3f} (목표 >= {TARGET_MAP50}) ===")
    print("클래스별 Recall:")

    very_high_recalls = []
    for pos, class_id in enumerate(class_ids):
        label = names[int(class_id)]
        recall = float(recalls[pos])
        flag = "  *** veryHigh ***" if label in VERY_HIGH_CLASSES else ""
        print(f"  {label:16s}: recall={recall:.3f}{flag}")
        if label in VERY_HIGH_CLASSES:
            very_high_recalls.append(recall)

    # 평가 데이터에 아예 없는 클래스는 지표가 나오지 않는다 — 조용히 넘어가면
    # "recall 목표 달성"으로 착각하게 되므로 명시한다.
    missing = [names[i] for i in sorted(names) if i not in {int(c) for c in class_ids}]
    if missing:
        print(f"\n평가 데이터에 인스턴스가 없어 지표를 못 낸 클래스: {', '.join(missing)}")

    if very_high_recalls:
        avg_very_high = sum(very_high_recalls) / len(very_high_recalls)
        status = "OK" if avg_very_high >= TARGET_VERY_HIGH_RECALL else "FAIL"
        print(f"\nveryHigh 평균 recall = {avg_very_high:.3f} "
              f"(목표 >= {TARGET_VERY_HIGH_RECALL}) [{status}]")
        missing_vh = VERY_HIGH_CLASSES - {names[int(c)] for c in class_ids}
        if missing_vh:
            print(f"  ! 단, {', '.join(sorted(missing_vh))} 는 평가에 포함되지 않았다 "
                  f"(데이터 없음). 위 평균은 나머지 클래스만의 값이다.")


if __name__ == "__main__":
    main()

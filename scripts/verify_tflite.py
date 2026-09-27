"""TFLite 모델이 docs/MODEL_INTERFACE.md 의 계약대로 동작하는지 검증한다.

Flutter 쪽에서 구현할 파이프라인을 파이썬으로 **그대로** 재현한다.
앱 결과가 이상할 때 "모델이 이상한가, 앱 코드가 이상한가"를 가르는 기준점이 된다.

  letterbox -> 양자화 입력 -> 추론 -> dequantize -> [1, C, 8400] 전치 디코드
  -> 클래스별 conf 임계값 -> 클래스별 NMS(IoU 0.45) -> letterbox 역변환

사용법:
    python scripts/verify_tflite.py --model stub_models/yolov8n_int8.tflite --image some.jpg
    python scripts/verify_tflite.py --model stub_models/yolov8n_int8.tflite \
        --image fixtures/images/fixture_01.jpg --coco --save out.jpg

  --coco    : 80-class COCO 모델일 때 COCO 라벨로 출력 (기본은 configs/labels.txt)
  --save    : 박스를 그린 이미지를 저장

주의: ai_edge_litert 는 비ASCII 경로의 파일을 열지 못한다. 이 스크립트는 모델을
      ASCII 임시 경로로 복사한 뒤 로드한다 (프로젝트 경로에 한글이 있어서 필요하다).
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
NMS_IOU = 0.45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TFLite I/O 계약 검증")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--coco", action="store_true", help="COCO 80-class 라벨 사용")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--conf", type=float, default=None,
                        help="클래스별 임계값 대신 이 값을 일괄 적용 (디코드 경로 눈으로 확인할 때)")
    return parser.parse_args()


def load_class_config(use_coco: bool) -> tuple[list[str], dict[int, float]]:
    """(라벨 목록, class_id -> conf 임계값) 반환."""
    if use_coco:
        import ultralytics

        p = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco8.yaml"
        names = yaml.safe_load(p.read_text(encoding="utf-8"))["names"]
        labels = [names[i] for i in sorted(names)]
        return labels, {i: 0.35 for i in range(len(labels))}

    cfg = yaml.safe_load((ROOT / "configs" / "class_labels.yaml").read_text(encoding="utf-8"))
    classes = sorted(cfg["classes"], key=lambda c: c["id"])
    return [c["label"] for c in classes], {c["id"]: c["conf_threshold"] for c in classes}


def letterbox(img: Image.Image, size: int) -> tuple[np.ndarray, float, float, float]:
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (int(pad_x), int(pad_y)))
    return np.asarray(canvas, dtype=np.float32) / 255.0, scale, pad_x, pad_y


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def main() -> None:
    args = parse_args()
    from ai_edge_litert.interpreter import Interpreter

    labels, thresholds = load_class_config(args.coco)

    # ai_edge_litert 는 비ASCII 경로를 못 연다 -> ASCII 임시 경로로 복사해서 로드한다.
    with tempfile.TemporaryDirectory(prefix="verify_") as tmp:
        tmp_model = Path(tmp) / "model.tflite"
        shutil.copy2(args.model, tmp_model)
        interpreter = Interpreter(model_path=str(tmp_model))
        interpreter.allocate_tensors()

        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]

        print("=" * 68)
        print("  텐서 정보 (docs/MODEL_INTERFACE.md §2, §3.1 대조용)")
        print("=" * 68)
        print(f"  입력  shape={list(inp['shape'])}  dtype={np.dtype(inp['dtype']).name}")
        print(f"        quant scale={inp['quantization'][0]:.8f} zero_point={inp['quantization'][1]}")
        print(f"  출력  shape={list(out['shape'])}  dtype={np.dtype(out['dtype']).name}")
        print(f"        quant scale={out['quantization'][0]:.8f} zero_point={out['quantization'][1]}")

        out_shape = list(out["shape"])
        num_classes = out_shape[1] - 4
        anchors = out_shape[2]
        print(f"  -> numClasses = shape[1] - 4 = {num_classes},  anchors = {anchors}")
        if num_classes != len(labels):
            print(f"  !! 라벨 개수({len(labels)}) != 모델 클래스 수({num_classes})")
        print("")

        img = Image.open(args.image)
        src_w, src_h = img.size
        tensor, scale, pad_x, pad_y = letterbox(img, args.imgsz)
        print(f"  이미지 {src_w}x{src_h} -> letterbox scale={scale:.4f} padX={pad_x} padY={pad_y}")

        # 입력 양자화 (앱의 §2.2 와 동일)
        in_scale, in_zero = inp["quantization"]
        if np.dtype(inp["dtype"]).kind in "iu" and in_scale != 0:
            data = np.round(tensor / in_scale) + in_zero
            info = np.iinfo(inp["dtype"])
            data = np.clip(data, info.min, info.max).astype(inp["dtype"])
        else:
            data = tensor.astype(inp["dtype"])

        interpreter.set_tensor(inp["index"], data[None, ...])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0]  # [C, anchors]

    # dequantize (§3.2)
    out_scale, out_zero = out["quantization"]
    if np.dtype(out["dtype"]).kind in "iu" and out_scale != 0:
        pred = (raw.astype(np.float32) - out_zero) * out_scale
    else:
        pred = raw.astype(np.float32)

    scores_all = pred[4:]
    print(f"  클래스 점수 범위: min={scores_all.min():.4f} max={scores_all.max():.4f}"
          f"  (0~1 이어야 정상 — §3.6 체크 2)")
    print(f"  bbox 행 범위    : min={pred[:4].min():.4f} max={pred[:4].max():.4f}"
          f"  (0~1 정규화 — §3.1)")
    print("")

    # 후보 추출 (§3.3)
    best_id = scores_all.argmax(axis=0)
    best_score = scores_all.max(axis=0)
    if args.conf is not None:
        thr = np.full(num_classes, args.conf)
    else:
        thr = np.array([thresholds.get(int(c), 0.35) for c in range(num_classes)])
    keep = best_score >= thr[best_id]

    cands = []
    for i in np.flatnonzero(keep):
        cx, cy, w, h = pred[0, i], pred[1, i], pred[2, i], pred[3, i]
        # letterbox 역변환 (§3.4)
        cx = (cx * args.imgsz - pad_x) / scale
        cy = (cy * args.imgsz - pad_y) / scale
        w = w * args.imgsz / scale
        h = h * args.imgsz / scale
        box = np.array([
            max(0.0, cx - w / 2), max(0.0, cy - h / 2),
            min(float(src_w), cx + w / 2), min(float(src_h), cy + h / 2),
        ])
        cands.append((float(best_score[i]), int(best_id[i]), box))

    print(f"  임계값 통과 후보: {len(cands)}개")

    # 클래스별 NMS (§3.5)
    cands.sort(key=lambda c: c[0], reverse=True)
    final = []
    for score, cid, box in cands:
        if any(c2 == cid and iou(box, b2) > NMS_IOU for _, c2, b2 in final):
            continue
        final.append((score, cid, box))

    print(f"  NMS(IoU={NMS_IOU}) 후 최종: {len(final)}개")
    print("")
    print("  최종 탐지 (상위 %d개)" % args.topk)
    print("  " + "-" * 64)
    for score, cid, box in final[: args.topk]:
        name = labels[cid] if cid < len(labels) else f"id{cid}"
        print(f"    {name:16s} conf={score:.3f}  "
              f"[{box[0]:7.1f}, {box[1]:7.1f}, {box[2]:7.1f}, {box[3]:7.1f}]")

    if args.save:
        vis = img.convert("RGB")
        draw = ImageDraw.Draw(vis)
        for score, cid, box in final:
            name = labels[cid] if cid < len(labels) else f"id{cid}"
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=5)
            draw.text((box[0] + 6, box[1] + 6), f"{name} {score:.2f}", fill=(255, 255, 0))
        args.save.parent.mkdir(parents=True, exist_ok=True)
        vis.save(args.save)
        print("")
        print(f"  시각화 저장: {args.save}")


if __name__ == "__main__":
    main()

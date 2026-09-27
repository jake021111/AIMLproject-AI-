"""배포 모델이 테스트 픽스처에서 내는 **실제 출력**을 JSON 으로 남긴다.

프론트가 앱에서 같은 이미지를 넣었을 때 같은 숫자가 나오는지 대조하는 기준값이다.
fixtures/expected_detections.json 은 '사람이 라벨링한 정답'이고, 이 파일은 '모델이 실제로 내는 값'이라
성격이 다르다. 앱 연동이 맞는지 보려면 **이 파일과 비교해야 한다.**

verify_tflite.py 와 동일한 전처리·디코드·NMS 를 쓴다.

사용법:
    python scripts/make_reference_outputs.py \
        --model release/insight_indoor15_v0.1_int8.tflite \
        --out fixtures/model_outputs_v0.1.json
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
NMS_IOU = 0.45


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="모델 실제 출력 기준값 생성")
    p.add_argument("--model", type=Path,
                   default=ROOT / "release" / "insight_indoor15_v0.1_int8.tflite")
    p.add_argument("--images", type=Path, default=ROOT / "fixtures" / "images")
    p.add_argument("--out", type=Path,
                   default=ROOT / "fixtures" / "model_outputs_v0.1.json")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def load_classes():
    cfg = yaml.safe_load((ROOT / "configs" / "class_labels.yaml").read_text(encoding="utf-8"))
    classes = sorted(cfg["classes"], key=lambda c: c["id"])
    return ([c["label"] for c in classes],
            {c["id"]: c["conf_threshold"] for c in classes})


def letterbox(img: Image.Image, size: int):
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR),
                 (int(pad_x), int(pad_y)))
    return np.asarray(canvas, dtype=np.float32) / 255.0, scale, pad_x, pad_y


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main() -> None:
    args = parse_args()
    from ai_edge_litert.interpreter import Interpreter

    labels, thresholds = load_classes()

    # ai_edge_litert 는 비ASCII 경로를 못 연다 → ASCII 임시 경로로 복사해 로드한다.
    with tempfile.TemporaryDirectory(prefix="refout_") as tmp:
        tmp_model = Path(tmp) / "m.tflite"
        shutil.copy2(args.model, tmp_model)
        interp = Interpreter(model_path=str(tmp_model))
        interp.allocate_tensors()
        inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
        num_classes = int(out["shape"][1]) - 4

        result = {
            "note": "배포 모델이 실제로 내는 출력. 앱 연동 검증용 기준값.",
            "model": args.model.name,
            "model_sha_prefix": None,
            "preprocess": {
                "input_shape": [int(v) for v in inp["shape"]],
                "layout": "NHWC",
                "color": "RGB",
                "resize": "letterbox (비율 유지, 회색 114 패딩)",
                "value_range": "0.0~1.0 (255로 나눔)",
            },
            "postprocess": {
                "output_shape": [int(v) for v in out["shape"]],
                "transposed": True,
                "conf_thresholds": {int(k): float(v) for k, v in thresholds.items()},
                "nms_iou": NMS_IOU,
                "nms_scope": "클래스별 (다른 클래스끼리는 억제하지 않음)",
            },
            "images": [],
        }

        import hashlib
        result["model_sha_prefix"] = hashlib.sha256(
            args.model.read_bytes()).hexdigest()[:16]

        for img_path in sorted(args.images.glob("*.jpg")):
            img = Image.open(img_path)
            src_w, src_h = img.size
            tensor, scale, pad_x, pad_y = letterbox(img, args.imgsz)

            data = tensor.astype(inp["dtype"])
            in_shape = list(inp["shape"])
            batched = data[None, ...]
            if in_shape[1] == 3 and in_shape[-1] != 3:  # NCHW 모델 대비
                batched = batched.transpose(0, 3, 1, 2)
            interp.set_tensor(inp["index"], batched)
            interp.invoke()
            pred = interp.get_tensor(out["index"])[0].astype(np.float32)

            scores = pred[4:]
            best_id = scores.argmax(axis=0)
            best_score = scores.max(axis=0)
            thr = np.array([thresholds.get(int(c), 0.35) for c in range(num_classes)])
            keep = best_score >= thr[best_id]

            cands = []
            for i in np.flatnonzero(keep):
                cx = pred[0, i] * args.imgsz
                cy = pred[1, i] * args.imgsz
                w = pred[2, i] * args.imgsz
                h = pred[3, i] * args.imgsz
                cx = (cx - pad_x) / scale
                cy = (cy - pad_y) / scale
                w, h = w / scale, h / scale
                box = (
                    max(0.0, cx - w / 2), max(0.0, cy - h / 2),
                    min(float(src_w), cx + w / 2), min(float(src_h), cy + h / 2),
                )
                cands.append((float(best_score[i]), int(best_id[i]), box))

            cands.sort(key=lambda c: c[0], reverse=True)
            final = []
            for score, cid, box in cands:
                if any(c2 == cid and iou(box, b2) > NMS_IOU for _, c2, b2 in final):
                    continue
                final.append((score, cid, box))

            result["images"].append({
                "file": img_path.name,
                "width": src_w,
                "height": src_h,
                "letterbox": {"scale": round(float(scale), 6),
                              "pad_x": round(float(pad_x), 2),
                              "pad_y": round(float(pad_y), 2)},
                "detections": [
                    {
                        "class_id": int(cid),
                        "label": labels[cid],
                        "confidence": round(float(score), 4),
                        "bbox_pixel": {
                            "left": round(float(b[0]), 1), "top": round(float(b[1]), 1),
                            "right": round(float(b[2]), 1), "bottom": round(float(b[3]), 1),
                        },
                    }
                    for score, cid, b in final
                ],
            })
            names = ", ".join(f"{labels[c]} {s:.2f}" for s, c, _ in final) or "(없음)"
            print(f"  {img_path.name}: {len(final)}개 — {names}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()

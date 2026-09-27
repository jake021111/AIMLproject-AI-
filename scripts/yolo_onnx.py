"""YOLOv8 -> ONNX 내보내기 (bbox 정규화 포함).

왜 별도 모듈인가:
    Ultralytics 의 raw 출력은 `[1, 4+nc, 8400]` 한 텐서에 bbox(letterbox 640 **픽셀**, 0~640)와
    클래스 점수(0~1)가 섞여 있다. TFLite INT8 은 per-tensor 양자화라 이 상태로 변환하면
    scale 이 640 쪽에 맞춰지고 클래스 점수가 전부 0 으로 뭉개진다.

    그래서 bbox 를 imgsz 로 나눠 0~1 로 맞춰야 하는데, 그 나눗셈을 **어디서** 하느냐가 중요하다.

      TF 쪽에서 SavedModel 을 감싸서 나누면  -> 변환은 되지만 INT8 캘리브레이션이
          감싼 함수 내부 텐서의 통계를 제대로 잡지 못해 bbox 가 전부 0 으로 무너진다 (실측).
      **ONNX 그래프 안에 넣으면**          -> 이후 단계가 평범한 단일 모델이라
          대표 데이터셋 캘리브레이션이 정상 동작한다.  <- 이 모듈이 하는 일

    Ultralytics 의 공식 TFLite export 도 같은 위치에서 정규화한다. 덕분에
    docs/MODEL_INTERFACE.md §3.1 의 "bbox 0~1 정규화" 계약이 플랫폼과 무관하게 유지된다.
"""
from pathlib import Path

import torch


class NormalizedDetect(torch.nn.Module):
    """YOLOv8 detect 모델의 bbox 출력을 0~1 로 정규화해 내보내는 래퍼."""

    def __init__(self, model: torch.nn.Module, imgsz: int) -> None:
        super().__init__()
        self.model = model
        self.imgsz = float(imgsz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        # ultralytics 는 (preds, extra) 튜플을 주기도 한다.
        if isinstance(y, (list, tuple)):
            y = y[0]
        # y: [B, 4+nc, 8400]  — 앞 4행이 cx, cy, w, h (픽셀)
        box = y[:, :4, :] / self.imgsz
        cls = y[:, 4:, :]
        return torch.cat([box, cls], dim=1)


def export_normalized_onnx(
    yolo_model,
    out_path: Path,
    imgsz: int = 640,
    opset: int = 13,
) -> Path:
    """ultralytics YOLO 객체를 bbox 정규화된 ONNX 로 내보낸다.

    Args:
        yolo_model: `ultralytics.YOLO` 인스턴스
        out_path:   저장할 .onnx 경로
        imgsz:      정사각 입력 크기 (letterbox 기준)
        opset:      ONNX opset

    Returns:
        저장된 onnx 경로
    """
    inner = yolo_model.model
    inner.eval()
    inner.float()
    # 추론 그래프로 고정한다 (export 모드가 아니면 학습용 분기가 섞인다).
    for module in inner.modules():
        if hasattr(module, "export"):
            module.export = True
        if hasattr(module, "format"):
            module.format = "onnx"

    wrapper = NormalizedDetect(inner, imgsz).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)

    with torch.no_grad():
        wrapper(dummy)  # 트레이스 전에 한 번 돌려 lazy 초기화를 끝낸다

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        opset_version=opset,
        input_names=["images"],
        output_names=["output0"],
        do_constant_folding=True,
        dynamo=False,
    )

    # 그래프 단순화는 선택이 아니다.
    # 정리하지 않은 torch.onnx.export 결과는 onnx2tf 의 SavedModel 변환에서
    # Reshape shape 추론에 실패한다:
    #   "Cannot reshape a tensor with 409600 elements to shape [1,64,1] ... node Reshape_48"
    # (Ultralytics 자체 export 는 내부적으로 onnxslim 을 돌리기 때문에 이 문제가 안 보인다.)
    try:
        import onnx
        import onnxslim

        onnx.save(onnxslim.slim(str(out_path)), str(out_path))
    except ImportError:
        print("경고: onnxslim 이 없어 그래프 단순화를 건너뛴다. "
              "onnx2tf 변환이 Reshape 에러로 실패할 수 있다.")

    return out_path

from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import io, json
from PIL import Image
import torch

# --- 너가 만든 모듈들 ---
from .detector import load_yolo, detect_and_crop
from .preprocessor import preprocess
from .classifier import load_classifier, predict as clf_predict, set_entropy_threshold as clf_set_Hth

# ====== 고정 설정 ======
ASSETS_DIR      = Path(__file__).resolve().parent.parent / "assets"
YOLO_WEIGHTS    = "yolov8s.pt"      # 또는 "yolov8n.pt"
CLF_WEIGHTS     = str(ASSETS_DIR / "classifier_best_v1.pt")
KO_MAPPING_CSV  = str(ASSETS_DIR / "ko_mapping.csv")   # ← 이것만 있으면 됨

MODEL_NAME      = "tf_efficientnetv2_m_in21ft1k"
NUM_CLASSES     = 105

INPUT_SIZE      = 480
PAD_RATIO       = 0.16
ENTROPY_TH_INIT = 2.0

_STATE: Dict[str, Any] = {"ready": False}

def _to_pil(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        img = x
    elif isinstance(x, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(x))
    elif isinstance(x, (str, Path)):
        img = Image.open(x)
    else:
        from PIL import Image as _Image
        img = _Image.fromarray(x)
    return img.convert("RGB")

def _load_once() -> None:
    if _STATE.get("ready"):
        return
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _STATE["device"] = device

    # 1) YOLO
    _STATE["yolo"] = load_yolo(
        weights=YOLO_WEIGHTS,
        conf=0.30,
        iou=0.50,
        imgsz=640,
        device=device
    )

    # 2) 분류기 (labels_path 제거, ko_mapping.csv만 사용)
    load_classifier(
        weight_path=CLF_WEIGHTS,
        labels_path=None,                # ← 이제 필요 없음
        ko_mapping_csv=KO_MAPPING_CSV,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        entropy_threshold=ENTROPY_TH_INIT,
        device=device,
        channels_last=True,
    )
    _STATE["ready"] = True

def _square_pad_whole(img: Image.Image, pad_ratio: float = PAD_RATIO) -> Image.Image:
    w, h = img.size
    side = int(round(max(w, h) * (1.0 + 2.0 * pad_ratio)))
    side = max(side, 1)
    left = (side - w) // 2
    top  = (side - h) // 2
    canvas = Image.new("RGB", (side, side))
    canvas.paste(img, (left, top))
    return canvas

def set_entropy_threshold(th: float) -> None:
    clf_set_Hth(float(th))

def predict(image: Any, return_topk: int = 3) -> Dict[str, Any]:
    _load_once()
    device = _STATE["device"]
    yolo   = _STATE["yolo"]

    img = _to_pil(image)

    det = detect_and_crop(img, pad_ratio=PAD_RATIO, pad_mode="edge", strategy="largest")
    crop = det["crop"] if det.get("crop") is not None else _square_pad_whole(img, PAD_RATIO)
    bbox = det.get("bbox")
    conf = det.get("conf")
    yolo_time = det.get("time_ms")
    W, H = (det.get("image_size") or img.size)

    x = preprocess(
        crop_img=crop,
        size=INPUT_SIZE,
        device=device,
        channels_last=True,
        add_batch=True,
    )

    out = clf_predict(x, topk=return_topk)

    result: Dict[str, Any] = {
        "prediction": {
            "decision": out["decision"],
            "decision_type": out["decision_type"],
            "top1": out["top1"],
            "topk": out["topk"],
            "reasons": {
                "D_entropy": {
                    "trigger": (out["decision_type"] == "mixed"),
                    "H": out["entropy"]["H"],
                    "threshold": out["entropy"]["threshold"]
                }
            }
        },
        "boxes": {
            "selected": bbox,
            "conf": conf,
            "detected": bbox is not None,
            "yolo_time_ms": yolo_time,
            "image_size": W if isinstance(W, (list, tuple)) else [W, H]
        },
        "meta": {
            "pad": PAD_RATIO,
            "input_size": INPUT_SIZE,
            "model": MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "device": device
        }
    }
    return result

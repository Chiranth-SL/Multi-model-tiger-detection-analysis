import gc
import io
import os
import time
import tracemalloc
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI(title="ResNet18 Tiger Detection Service")

CLASS_NAMES = {0: "no_tiger", 1: "tiger"}
WEIGHTS_PATH = "/weights/ResNet18_tiger.pt"
LOCAL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "ResNet18_tiger.pt")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_weights_path() -> str:
    if os.path.exists(WEIGHTS_PATH):
        return WEIGHTS_PATH
    if os.path.exists(LOCAL_WEIGHTS_PATH):
        return LOCAL_WEIGHTS_PATH
    raise FileNotFoundError(f"Weights file not found at {WEIGHTS_PATH} or {LOCAL_WEIGHTS_PATH}")


@app.get("/")
def read_root():
    return {"service": "ResNet18 Tiger Detection", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "ResNet18"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        # Still attempt to open if content-type header is generic or missing, but validate with PIL
        pass

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    model = None
    input_tensor = None
    outputs = None

    # Start memory and latency profiling
    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        # Load architecture
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)

        # Load weights on CPU
        weights_file = get_weights_path()
        state_dict = torch.load(weights_file, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        elif isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        elif isinstance(state_dict, nn.Module):
            model = state_dict
        else:
            raise ValueError("Unsupported checkpoint format")

        model.eval()

        # Preprocess and infer
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probabilities).item()
            confidence = probabilities[pred_idx].item() * 100.0

        end_time = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        inference_time_ms = round((end_time - start_time) * 1000, 2)
        memory_used_mb = round(peak_mem / (1024 * 1024), 2)

        pred_label = CLASS_NAMES.get(pred_idx, "unknown")
        has_tiger = bool(pred_idx == 1)

        return {
            "model_name": "ResNet18",
            "prediction": pred_label,
            "has_tiger": has_tiger,
            "confidence_percentage": round(confidence, 2),
            "confidence": round(confidence, 2),
            "inference_time_ms": inference_time_ms,
            "memory_used_mb": memory_used_mb
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:
        tracemalloc.stop()
        if model is not None:
            del model
        if input_tensor is not None:
            del input_tensor
        if outputs is not None:
            del outputs
        gc.collect()

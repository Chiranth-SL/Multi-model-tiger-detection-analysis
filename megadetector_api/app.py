import gc
import io
import os
import time
import tracemalloc
import builtins
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Bulletproof fix for ANY package asking interactive y/n questions in Docker (like YOLO telemetry/conversion)
builtins.input = lambda prompt='': 'n'

app = FastAPI(title="MegaDetector v5 API", version="1.0.0")

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "/app/md_v5a.0.0.pt")


def load_model(weights_path: str):
    """Loads MegaDetector v5 model using torch.hub on CPU."""
    import torch

    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=weights_path,
        force_reload=False,
        trust_repo=True,
        device="cpu",
    )
    return ("yolov5_hub", model)


@app.get("/")
def root():
    return {"status": "ok", "service": "MegaDetector v5 API"}


@app.get("/health")
def health():
    weights_found = os.path.exists(WEIGHTS_PATH)
    return {
        "status": "ok" if weights_found else "warning",
        "model": "MegaDetector_v5",
        "weights_path": WEIGHTS_PATH,
        "weights_found": weights_found,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate image upload
    is_image = (
        (file.content_type and file.content_type.startswith("image/"))
        or (file.filename and file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
    )
    if not is_image:
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

    if not os.path.exists(WEIGHTS_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Model weights not found at path: {WEIGHTS_PATH}",
        )

    # Start memory tracing
    tracemalloc.start()
    start_time = time.perf_counter()

    model_type = None
    model = None
    has_tiger = False
    highest_conf = 0.0

    try:
        model_type, model = load_model(WEIGHTS_PATH)

        if model_type == "ultralytics":
            results = model.predict(source=image, device="cpu", conf=0.25, verbose=False)
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()
                    # MegaDetector classes: 0 = animal, 1 = person, 2 = vehicle
                    animal_confs = [
                        float(c)
                        for c, cls_id in zip(confs, classes)
                        if int(cls_id) == 0 and float(c) > 0.25
                    ]
                    if animal_confs:
                        has_tiger = True
                        highest_conf = max(animal_confs)
        else:
            # YOLOv5 hub model inference
            results = model(image)
            # Pred format: [xmin, ymin, xmax, ymax, conf, class]
            preds = results.xyxy[0].cpu().numpy()
            if len(preds) > 0:
                animal_confs = [
                    float(row[4])
                    for row in preds
                    if int(row[5]) == 0 and float(row[4]) > 0.25
                ]
                if animal_confs:
                    has_tiger = True
                    highest_conf = max(animal_confs)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        # Explicitly delete model and garbage collect
        if model is not None:
            del model
        gc.collect()

    end_time = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    inference_time_ms = round((end_time - start_time) * 1000, 2)
    memory_used_mb = round(peak_memory / (1024 * 1024), 2)
    confidence_pct = round(highest_conf * 100, 2) if has_tiger else 0.0
    prediction = "tiger" if has_tiger else "no_tiger"

    return JSONResponse(
        content={
            "model": "MegaDetector_v5",
            "model_name": "MegaDetector_v5",
            "prediction": prediction,
            "has_tiger": has_tiger,
            "confidence": confidence_pct,
            "inference_time_ms": inference_time_ms,
            "memory_used_mb": memory_used_mb,
        }
    )

import gc
import io
import os
import time
import tracemalloc
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="YOLOv8n Tiger Detection API", version="1.0.0")

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "/weights/yolov8n_tiger.pt")


@app.get("/")
def root():
    return {"status": "ok", "service": "YOLOv8n Tiger Detection API"}


@app.get("/health")
def health():
    weights_found = os.path.exists(WEIGHTS_PATH)
    return {
        "status": "ok" if weights_found else "warning",
        "model": "YOLOv8n",
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

    model = None
    has_tiger = False
    highest_conf = 0.0

    try:
        # Load model on CPU only for request lifecycle
        model = YOLO(WEIGHTS_PATH)

        # Run inference on CPU
        results = model.predict(source=image, device="cpu", conf=0.25, verbose=False)

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                valid_confs = [float(c) for c in confs if float(c) > 0.25]
                if valid_confs:
                    has_tiger = True
                    highest_conf = max(valid_confs)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        # Explicitly unload model and invoke garbage collection
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
            "model": "YOLOv8n",
            "model_name": "YOLOv8n",
            "prediction": prediction,
            "has_tiger": has_tiger,
            "confidence": confidence_pct,
            "inference_time_ms": inference_time_ms,
            "memory_used_mb": memory_used_mb,
        }
    )

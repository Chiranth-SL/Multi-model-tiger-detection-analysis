# Tiger Detection Benchmarking 🐅

A highly optimized, Dockerized microservices architecture designed to evaluate and benchmark 6 different computer vision models on tiger detection. 

This project allows you to run multiple heavy deep-learning models simultaneously on standard consumer hardware (e.g., 16GB RAM laptops) by utilizing a smart, sequential load/unload memory management system.

## 🏗️ Architecture

The application is built using a **Microservices Architecture**:
- **Input/Output (Web UI):** A Streamlit dashboard (`web_ui/`)
- **Model APIs:** 6 independent FastAPI containers, each hosting a different architecture:
  1. **YOLOv8 Nano** (Native Object Detection)
  2. **MegaDetector v5** (Wildlife YOLOv5 Specialist)
  3. **ResNet18** (Image Classification)
  4. **VGG11** (Deep Image Classification)
  5. **GoogLeNet** (Inception Classification)
  6. **SpeciesNet** (Species Classification)

## 🚀 Getting Started

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### 1. Model Weights (Important!)
Because GitHub restricts files larger than 100MB, the model weights must be placed in the `weights/` directory before running. 
Ensure your `weights/` folder contains:
- `yolov8n_tiger.pt`
- `ResNet18_tiger.pt`
- `VGG11_tiger.pt`
- `GoogLeNet_tiger.pt`

*(Note: MegaDetector and SpeciesNet automatically download their weights during the Docker build process).*

### 2. Start the Containers
Open your terminal in the `docker_app` folder (where the `docker-compose.yml` file is located) and run:
```bash
docker-compose up --build
```
*(Note: The `--build` flag is only required the very first time you run the application, or if you make changes to the code. For everyday use, you can skip the build step and just run `docker-compose up` to start it much faster. Add `-d` to run it in the background: `docker-compose up -d`)*

### 3. Access the Dashboard
Once the containers have started, open your web browser and navigate to:
👉 **http://localhost:8501**

Upload a test image, click **Run All Models**, and view the real-time latency, memory usage, and confidence comparisons!

### 4. Stop the Containers
When you are finished using the application, it's important to shut down the containers to free up your computer's resources. 

If you are running the containers in the foreground, simply press `Ctrl+C` in your terminal.

If you ran them in the background (using `-d`), or just want to ensure everything is completely cleaned up, run the following command in the same directory:
```bash
docker-compose down
```

## 🧠 Smart Memory Management
Running 6 deep learning models typically causes CUDA Out-Of-Memory (OOM) crashes on standard laptops. This repository solves this by:
1. **Idle State:** Containers idle at <50MB of RAM.
2. **Sequential Invocation:** The Web UI pings the APIs one at a time.
3. **Lazy Loading & GC:** Models are loaded into memory *only* when the API receives a request, and are explicitly deleted (`del model`, `gc.collect()`) the millisecond inference is complete.

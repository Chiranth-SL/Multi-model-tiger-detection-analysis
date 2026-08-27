import io
import time
import requests
import pandas as pd
from PIL import Image
import streamlit as st
import altair as alt

# Page Configuration
st.set_page_config(
    page_title="Tiger Detection Model Benchmarking",
    page_icon="🐅",
    layout="wide"
)

# Header Section
st.title("🐅 Tiger Detection Model Benchmarking")
st.markdown(
    "Upload a camera trap or wildlife image to benchmark and compare inference performance, "
    "accuracy, and memory utilization across multiple deep learning architectures."
)

# List of model API endpoints
MODEL_ENDPOINTS = [
    {"name": "ResNet-18", "url": "http://resnet_api:8000/predict", "type": "Classifier"},
    {"name": "VGG-11", "url": "http://vgg_api:8000/predict", "type": "Classifier"},
    {"name": "GoogLeNet", "url": "http://googlenet_api:8000/predict", "type": "Classifier"},
    {"name": "YOLOv8", "url": "http://yolo_api:8000/predict", "type": "Detector"},
    {"name": "MegaDetector", "url": "http://megadetector_api:8000/predict", "type": "Wildlife Detector"},
    {"name": "SpeciesNet", "url": "http://speciesnet_api:8000/predict", "type": "Species Classifier"},
]

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("### Benchmark Models")
    for endpoint in MODEL_ENDPOINTS:
        st.markdown(f"- **{endpoint['name']}** ({endpoint['type']})")
    
    st.divider()
    timeout_sec = st.slider("API Timeout (seconds)", min_value=5, max_value=60, value=30, step=5)
    st.info("Ensure all model containers are running on the Docker bridge network.")

# File Uploader
uploaded_file = st.file_uploader(
    "Choose an image for tiger detection...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Uploaded Image")
        try:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption=f"{uploaded_file.name} ({image.size[0]}x{image.size[1]})", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load image: {e}")

    with col2:
        st.subheader("🚀 Benchmarking Control")
        st.write("Click below to run sequential inference across all 6 model APIs.")
        run_benchmark = st.button("▶️ Run All Models", type="primary", use_container_width=True)

    if run_benchmark:
        st.divider()
        st.subheader("📊 Benchmark Results")

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, endpoint in enumerate(MODEL_ENDPOINTS):
            model_name = endpoint["name"]
            url = endpoint["url"]
            status_text.text(f"Running inference on {model_name} ({idx+1}/{len(MODEL_ENDPOINTS)})...")

            with st.spinner(f"Processing with {model_name}..."):
                try:
                    files = {
                        "file": (uploaded_file.name, image_bytes, uploaded_file.type or "image/jpeg")
                    }
                    response = requests.post(url, files=files, timeout=timeout_sec)

                    if response.status_code == 200:
                        data = response.json()
                        has_tiger = data.get("has_tiger", False)
                        confidence = float(data.get("confidence", 0.0))
                        inference_time = float(data.get("inference_time_ms", 0.0))
                        memory_used = float(data.get("memory_used_mb", 0.0))
                        
                        # Determine human-readable prediction status
                        prediction_status = "Tiger Detected" if has_tiger else "No Tiger"
                        
                        results.append({
                            "Model": model_name,
                            "Prediction": prediction_status,
                            "Confidence(%)": round(confidence, 2),
                            "Inference Time(ms)": round(inference_time, 2),
                            "Memory Used(MB)": round(memory_used, 2),
                            "Status": "Success",
                            "Raw_Prediction": data.get("prediction", "")
                        })
                    else:
                        st.error(f"❌ {model_name} API returned HTTP {response.status_code}: {response.text}")
                        results.append({
                            "Model": model_name,
                            "Prediction": "Error",
                            "Confidence(%)": 0.0,
                            "Inference Time(ms)": 0.0,
                            "Memory Used(MB)": 0.0,
                            "Status": f"HTTP {response.status_code}",
                            "Raw_Prediction": "API Error"
                        })

                except requests.exceptions.RequestException as req_err:
                    st.error(f"❌ Failed to connect to {model_name} ({url}): {str(req_err)}")
                    results.append({
                        "Model": model_name,
                        "Prediction": "Error",
                        "Confidence(%)": 0.0,
                        "Inference Time(ms)": 0.0,
                        "Memory Used(MB)": 0.0,
                        "Status": "Connection Failed",
                        "Raw_Prediction": "Network Error"
                    })
                except Exception as ex:
                    st.error(f"❌ Unexpected error for {model_name}: {str(ex)}")
                    results.append({
                        "Model": model_name,
                        "Prediction": "Error",
                        "Confidence(%)": 0.0,
                        "Inference Time(ms)": 0.0,
                        "Memory Used(MB)": 0.0,
                        "Status": "Exception",
                        "Raw_Prediction": "Internal Error"
                    })

            progress_bar.progress((idx + 1) / len(MODEL_ENDPOINTS))

        status_text.text("✅ Benchmark completed across all models!")
        time.sleep(0.5)

        if results:
            df = pd.DataFrame(results)

            # Display high-level summary KPI metrics
            successful_runs = df[df["Status"] == "Success"]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Models Evaluated", len(df))
            with m2:
                tiger_count = len(df[df["Prediction"] == "Tiger Detected"])
                st.metric("Tigers Detected", f"{tiger_count} / {len(df)}")
            with m3:
                avg_time = successful_runs["Inference Time(ms)"].mean() if not successful_runs.empty else 0.0
                st.metric("Avg Inference Time", f"{avg_time:.2f} ms")
            with m4:
                fastest_model = successful_runs.sort_values(by="Inference Time(ms)").iloc[0]["Model"] if not successful_runs.empty else "N/A"
                st.metric("Fastest Model", fastest_model)

            st.write("")

            # Define table columns
            display_cols = ["Model", "Prediction", "Confidence(%)", "Inference Time(ms)", "Memory Used(MB)"]
            display_df = df[display_cols]

            # Style function for color-coding predictions
            def highlight_prediction_cells(val):
                if val == "Tiger Detected":
                    return "background-color: #28a745; color: white; font-weight: bold;"
                elif val == "No Tiger":
                    return "background-color: #dc3545; color: white; font-weight: bold;"
                elif val == "Error":
                    return "background-color: #ffc107; color: black; font-weight: bold;"
                return ""

            styled_table = (
                display_df.style
                .map(highlight_prediction_cells, subset=["Prediction"])
                .format({
                    "Confidence(%)": "{:.2f}%",
                    "Inference Time(ms)": "{:.2f}",
                    "Memory Used(MB)": "{:.2f}"
                })
            )

            st.markdown("### 📋 Model Comparison Table")
            st.dataframe(
                styled_table,
                use_container_width=True,
                hide_index=True
            )

            # Charts section
            st.divider()
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### ⚡ Inference Time Comparison (ms)")
                valid_df = df[df["Inference Time(ms)"] > 0]
                if not valid_df.empty:
                    chart_time = (
                        alt.Chart(valid_df)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("Model:N", sort=None, title="Model Architecture"),
                            y=alt.Y("Inference Time(ms):Q", title="Inference Time (ms)"),
                            color=alt.Color(
                                "Prediction:N",
                                scale=alt.Scale(
                                    domain=["Tiger Detected", "No Tiger", "Error"],
                                    range=["#28a745", "#dc3545", "#ffc107"]
                                ),
                                title="Prediction"
                            ),
                            tooltip=["Model", "Prediction", "Inference Time(ms)", "Confidence(%)"]
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(chart_time, use_container_width=True)
                else:
                    st.info("No inference timing data available to display.")

            with c2:
                st.markdown("### 💾 Peak Memory Usage Comparison (MB)")
                valid_mem_df = df[df["Memory Used(MB)"] > 0]
                if not valid_mem_df.empty:
                    chart_mem = (
                        alt.Chart(valid_mem_df)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#4b6cb7")
                        .encode(
                            x=alt.X("Model:N", sort=None, title="Model Architecture"),
                            y=alt.Y("Memory Used(MB):Q", title="Memory Used (MB)"),
                            tooltip=["Model", "Memory Used(MB)", "Inference Time(ms)"]
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(chart_mem, use_container_width=True)
                else:
                    st.info("No memory usage data available to display.")

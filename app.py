import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import time

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="YOLOv8 Military Object Detection", page_icon="🛰️", layout="wide")
st.title("🛰️ Military Object Detection System")
st.markdown(
    """
    ### Real-time Object Detection using YOLOv8  
    Upload an **image** or **video** to detect objects like Aircraft, Helicopter, Tank, Truck, Person, and Weapon.
    """
)

# -------------------- Sidebar --------------------
st.sidebar.header("⚙️ Configuration")
model_path = st.sidebar.text_input(
    "Enter Model Path",
    value=r"C:\Users\nikhi\Downloads\object_detection\runs\military_yolov8l_split1\weights\best.pt"
)
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
device = st.sidebar.selectbox("Select Device", ["cuda", "cpu"])

# Load YOLO model
try:
    model = YOLO(model_path)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Model load failed: {e}")
    st.stop()

# -------------------- Upload Section --------------------
uploaded_file = st.file_uploader("📤 Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"])

if uploaded_file:
    # Save file temporarily
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"✅ File uploaded: {uploaded_file.name}")
    file_ext = uploaded_file.name.split(".")[-1].lower()

    # -------------------- IMAGE --------------------
    if file_ext in ["jpg", "jpeg", "png"]:
        st.image(tmp_path, caption="🖼️ Uploaded Image", use_column_width=True)
        st.info("🔍 Detecting objects...")

        results = model.predict(source=tmp_path, conf=conf, device=device, save=True)
        output_dir = results[0].save_dir
        output_path = os.path.join(output_dir, uploaded_file.name)

        if os.path.exists(output_path):
            st.success("✅ Detection complete!")
            st.image(output_path, caption="📸 Detected Image", use_column_width=True)
        else:
            st.error("⚠️ Processed image not found!")

    # -------------------- VIDEO --------------------
    elif file_ext in ["mp4", "mov", "avi", "mkv"]:
        st.video(tmp_path)
        st.info("🎬 Processing video... Please wait, this might take a few minutes.")

        start_time = time.time()
        results = model.predict(source=tmp_path, conf=conf, device=device, save=True)
        end_time = time.time()

        output_dir = results[0].save_dir
        output_video = None

        # Wait for YOLO to finish writing the output file
        for i in range(20):  # wait up to ~10 seconds
            for f in os.listdir(output_dir):
                if f.endswith((".mp4", ".avi", ".mov", ".mkv")):
                    output_video = os.path.join(output_dir, f)
                    break
            if output_video and os.path.exists(output_video):
                break
            time.sleep(0.5)

        if output_video and os.path.exists(output_video):
            st.success(f"✅ Detection completed in {end_time - start_time:.2f}s")
            st.video(output_video)
        else:
            st.error("⚠️ Processed video not found. Try again or check YOLO output folder manually.")

st.markdown("---")
st.caption("💻 Developed by Nikhil | Powered by YOLOv8 + Streamlit")

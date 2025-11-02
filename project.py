import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import pyttsx3
import numpy as np
import time

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="♻️ Smart Waste Segregation System",
    page_icon="🗑️",
    layout="wide"
)

# =====================
# Load YOLO Model (Cached)
# =====================
@st.cache_resource
def load_model(path):
    return YOLO(path)

model_path = r"C:\Users\moham\OneDrive\Desktop\object\runs\detect\garbage_fast_quality\weights\best.pt"
model = load_model(model_path)

# =====================
# Text-to-Speech
# =====================
def speak(text):
    engine = pyttsx3.init("sapi5")
    engine.setProperty("rate", 170)
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[0].id)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# =====================
# Sidebar Settings
# =====================
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
iou_thresh = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.5, 0.05)
show_tts = st.sidebar.toggle("🔊 Enable Voice Feedback", True)

# =====================
# Header Section
# =====================
st.markdown(
    """
    <style>
        .main-title {
            text-align:center;
            font-size:42px;
            color:#2ecc71;
            font-weight:800;
        }
        .subtitle {
            text-align:center;
            font-size:20px;
            color:#555;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>♻️ Smart Waste Segregation System</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='subtitle'> Enhancing waste management through AI-powered classification</h4>", unsafe_allow_html=True)
st.markdown("---")

# =====================
# Info Section
# =====================
with st.expander("📘 How to Use", expanded=True):
    st.write("""
    1️⃣ Upload a clear image containing trash items.  
    2️⃣ Adjust confidence or IoU values from the sidebar if needed.  
    3️⃣ Click **Detect Trash** and view the results.  
    """)

# =====================
# File Upload
# =====================
uploaded_file = st.file_uploader("Upload a trash image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    # Resize uploaded image to a standard width (600px)
    max_width = 600
    img_width, img_height = img.size
    if img_width > max_width:
        ratio = max_width / img_width
        new_size = (int(img_width * ratio), int(img_height * ratio))
        img_resized = img.resize(new_size)
    else:
        img_resized = img

    st.image(img_resized, caption="Uploaded Image", use_container_width=False)

    if st.button(" Detect Trash"):
        with st.spinner("Analyzing image... please wait"):
            time.sleep(0.8)
            results = model.predict(
                source=np.array(img),
                conf=confidence,
                iou=iou_thresh,
                agnostic_nms=True,
                save=False,
                show=False
            )

        # =====================
        # Draw Detection Boxes
        # =====================
        detected_classes = []
        result_img = img.copy()
        draw = ImageDraw.Draw(result_img)

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = results[0].names[cls_id].lower()

            if class_name in ["biodegradable", "organic"]:
                class_name = "ORGANIC"
                color = (0, 200, 0)
            else:
                class_name = "NON-ORGANIC"
                color = (200, 0, 0)

            detected_classes.append(class_name)
            xy = box.xyxy[0].cpu().numpy()
            draw.rectangle([xy[0], xy[1], xy[2], xy[3]], outline=color, width=3)
            draw.text((xy[0]+5, xy[1]+5), class_name, fill=color)

        unique_classes = list(set(detected_classes))
        sentence = "Detected: " + ", ".join(unique_classes) if unique_classes else "No trash detected."

        # Resize result image for clean display
        max_width_result = 700
        img_width, img_height = result_img.size
        if img_width > max_width_result:
            ratio = max_width_result / img_width
            new_size = (int(img_width * ratio), int(img_height * ratio))
            result_img = result_img.resize(new_size)

        # =====================
        # Display Results
        # =====================
        st.markdown("## 🧾 Detection Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.image(result_img, caption="Detection Result", use_container_width=False)
        with col2:
            st.success(sentence)
            if show_tts:
                speak(sentence)

            st.markdown("### 📊 Trash Classification Summary")
            organic_count = detected_classes.count("ORGANIC")
            non_organic_count = detected_classes.count("NON-ORGANIC")

            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="♻️ ORGANIC", value=organic_count)
            with c2:
                st.metric(label="🧃 NON-ORGANIC", value=non_organic_count)

            total = organic_count + non_organic_count
            if total > 0:
                org_percent = (organic_count / total) * 100
                non_percent = (non_organic_count / total) * 100
                st.progress(int(org_percent))
                st.caption(f"Organic Trash: {org_percent:.1f}% | Non-Organic Trash: {non_percent:.1f}%")

# =====================
# Footer
# =====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#666; font-size:16px;'>
        Made with ❤️ by <strong>Mohamed Zuhair</strong>
                    
    </div>
    """,
    unsafe_allow_html=True
)

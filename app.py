import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model = load_model("mango_model_final.h5", compile=False)

# =========================
# CLASS LABEL
# =========================
class_names = [
    'Anthracnose',
    'Bacterial Canker',
    'Cutting Weevil',
    'Die Back',
    'Gall Midge',
    'Healthy',
    'Powdery Mildew',
    'Sooty Mould'
]

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon="🍃",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
body {
    background-color: #ffe6f2;
}

.main {
    background-color: #ffe6f2;
}

h1 {
    text-align: center;
    color: #ff1493;
    font-family: cursive;
    font-size: 45px;
}

.stButton>button {
    background-color: #ff69b4;
    color: white;
    border-radius: 15px;
    border: none;
    padding: 10px 25px;
    font-size: 18px;
}

.css-1d391kg {
    background-color: #fff0f5;
}

.result-box {
    background-color: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<h1>🍃 Mango Leaf Disease Detection 🍃</h1>", unsafe_allow_html=True)

st.write("### Upload gambar daun mangga untuk mendeteksi penyakit.")

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader(
    "Upload gambar daun",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    # preprocessing
    img_resize = img.resize((224, 224))

    img_array = np.array(img_resize) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction) * 100

    # result
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.subheader("Hasil Prediksi")

    if predicted_class == "Healthy":
        st.success(f"Daun Sehat ✅ ({confidence:.2f}%)")

    else:
        st.error(f"Penyakit: {predicted_class} ⚠️ ({confidence:.2f}%)")

    st.markdown("</div>", unsafe_allow_html=True)
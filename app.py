import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

st.title("Diagnosis Renal Cell Carcinoma")
st.subheader("Upload Gambar CT SCAN Ginjal")


# LOAD 3 MODEL
@st.cache_resource
def load_models():
    verify_model_path = os.path.join("model", "verify", "Model_verifikasi.h5")
    if not os.path.exists(verify_model_path):
        verify_model_path = os.path.join("model", "verify", "model_verifikasi.h5")

    verify_model = tf.keras.models.load_model(verify_model_path)
    cnn_model = tf.keras.models.load_model(os.path.join("model", "Model_CNN.h5"))
    resnet_model = tf.keras.models.load_model(os.path.join("model", "Model_ResNet50.h5"))
    return verify_model, cnn_model, resnet_model

verify_model, cnn_model, resnet_model = load_models()


# FUNGSI PREPROCESS UNTUK VERIFIKASI (Xception - 224x224)
def preprocess_verification(image_data):
    img = image_data.resize((256, 256))
    img_array = np.array(img, dtype=np.float32)
    img_array = xception_preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# FUNGSI PREPROCESS UNTUK DIAGNOSIS (256x256)
def preprocess_diagnosis(image_data):
    img = image_data.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# FUNGSI VERIFIKASI (Apakah ini CT Scan Ginjal)
def verify_image(model, image_data):
    img_array = preprocess_verification(image_data)
    prediction = model.predict(img_array, verbose=0)
    score = prediction[0][0]
    
    # Threshold binary: < 0.5 adalah kidney_ct, >= 0.5 adalah not_kidney_ct
    is_kidney_ct = score < 0.5
    confidence = (1 - score) if is_kidney_ct else score
    return is_kidney_ct, confidence


# FUNGSI DIAGNOSIS (Normal / Tumor)
def predict_diagnosis(model, image_data):
    img_array = preprocess_diagnosis(image_data)
    prediction = model.predict(img_array, verbose=0)

    label = "Normal" if prediction[0][0] > 0.5 else "Tumor"
    confidence = (
        prediction[0][0]
        if prediction[0][0] > 0.5
        else 1 - prediction[0][0]
    )

    return label, confidence


# UPLOAD GAMBAR
uploaded_file = st.file_uploader(
    "Upload Gambar CT Scan Ginjal",
    type=["jpg", "jpeg", "png"]
)

# PROSES FLOWCHART
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang Diupload",
        width="stretch"
    )

    # STEP 1: Verifikasi Gambar
    with st.spinner("Memverifikasi gambar..."):
        is_kidney_ct, verify_conf = verify_image(verify_model, image)

    if not is_kidney_ct:
        # Tolak Gambar jika bukan CT Scan Ginjal
        st.error("Gambar yang diupload bukan CT Scan Ginjal")
    else:
        st.success(f"Verifikasi Berhasil: Gambar terdeteksi sebagai CT Scan Ginjal")
        st.markdown("---")

        # STEP 2: CNN & ResNet50 Diagnosis
        with st.spinner("Menganalisis diagnosis..."):
            cnn_label, cnn_conf = predict_diagnosis(cnn_model, image)
            resnet_label, resnet_conf = predict_diagnosis(resnet_model, image)

        st.markdown("## Hasil Diagnosis")

        # CNN
        st.success(f"**CNN Model : {cnn_label}**")
        st.write(f"Confidence Score CNN : **{cnn_conf:.2%}**")

        st.markdown("---")

        # ResNet50
        st.success(f"**ResNet50 Model : {resnet_label}**")
        st.write(f"Confidence Score ResNet50 : **{resnet_conf:.2%}**")

# DISCLAIMER
st.markdown("---")
st.info(
    "**Disclaimer:** Hasil diagnosis ini bersifat informatif "
    "dan bertujuan sebagai pendukung keputusan medis. "
    "Konsultasikan hasil diagnosis anda kepada tenaga medis profesional."
)

import streamlit as st
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

st.set_page_config(
    page_title="Crop Disease Severity Estimator",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1f0f 100%);
    min-height: 100vh;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4ade80, #22d3ee, #4ade80);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite linear;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}

.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 300;
    margin-bottom: 2rem;
    letter-spacing: 0.02em;
}

.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
}

.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #4ade80;
}

.stat-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #4ade80;
    margin-bottom: 0.5rem;
}

.result-box {
    background: rgba(74,222,128,0.06);
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-number {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    color: #4ade80;
    line-height: 1;
}

.result-label {
    font-size: 0.85rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.5rem;
}

.severity-badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-size: 0.9rem;
    font-weight: 500;
    margin-top: 1rem;
    letter-spacing: 0.05em;
}

.low { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid #4ade80; }
.medium { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid #fbbf24; }
.high { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid #ef4444; }

.info-pill {
    background: rgba(34,211,238,0.08);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 50px;
    padding: 0.4rem 1rem;
    font-size: 0.78rem;
    color: #22d3ee;
    display: inline-block;
    margin: 0.2rem;
}

.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Style sliders */
.stSlider > div > div > div > div {
    background: #4ade80 !important;
}

/* Style selectbox */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(74,222,128,0.2) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Style button */
.stButton > button {
    background: linear-gradient(135deg, #4ade80, #22d3ee) !important;
    color: #0a1628 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(74,222,128,0.3) !important;
}

/* File uploader */
.stFileUploader > div {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(74,222,128,0.25) !important;
    border-radius: 16px !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_xgb = pickle.load(open('models/xgb_model.pkl', 'rb'))
    scaler    = pickle.load(open('models/scaler.pkl',    'rb'))
    return model_xgb, scaler

@st.cache_resource
def load_resnet():
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    resnet  = models.resnet50(weights=weights)
    resnet  = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    return resnet

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def extract_feature(image):
    resnet = load_resnet()
    img = image.convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feature = resnet(img_tensor)
    return feature.squeeze().numpy()

disease_list = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust",
    "Apple___healthy","Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot",
    "Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy","Tomato___Bacterial_spot",
    "Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ── HERO SECTION ───────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌿 Crop Disease<br>Severity Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ResNet-50 + XGBoost + SHAP · PlantVillage Dataset · 38 Disease Classes</div>', unsafe_allow_html=True)
# Stats row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-number">54K+</div><div class="stat-label">Training Images</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-number">90.6%</div><div class="stat-label">R² Accuracy</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-number">38</div><div class="stat-label">Disease Classes</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="stat-number">2052</div><div class="stat-label">Feature Dimensions</div></div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Supported crops pills
st.markdown('<div class="section-label">✅ Supported Crops</div>', unsafe_allow_html=True)
crops = ["Apple","Blueberry","Cherry","Corn","Grape","Orange","Peach","Pepper","Potato","Raspberry","Soybean","Squash","Strawberry","Tomato"]
pills_html = "".join([f'<span class="info-pill">{c}</span>' for c in crops])
st.markdown(pills_html, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── MAIN LAYOUT ────────────────────────────────────────────────────
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown('<div class="section-label">🔬 Disease Selection</div>', unsafe_allow_html=True)
    selected_disease = st.selectbox('', disease_list, label_visibility="collapsed")

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🖼️ Leaf Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader('', type=['jpg','jpeg','png'], label_visibility="collapsed")

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption='Uploaded leaf', width=400)

with right:
    st.markdown('<div class="section-label">🌤️ Weather Conditions</div>', unsafe_allow_html=True)

    wc1, wc2 = st.columns(2)
    with wc1:
        temp     = st.slider('🌡️ Temperature (°C)', 5,  45, 22)
        humidity = st.slider('💧 Humidity (%)',     20, 100, 70)
    with wc2:
        rainfall = st.slider('🌧️ Rainfall (mm)',    0,  50,  5)
        wind     = st.slider('💨 Wind (km/h)',       0,  60, 15)

    st.markdown('<br>', unsafe_allow_html=True)

    predict_btn = st.button('⚡ Analyse & Predict Yield Loss')

    if predict_btn:
        if uploaded is None:
            st.warning('Please upload a leaf image first!')
        else:
            with st.spinner('Running AI analysis...'):
                model_xgb, scaler = load_model()
                cnn_feat = extract_feature(image)
                weather  = np.array([[temp, humidity, rainfall, wind]])
                w_scaled = scaler.transform(weather)
                X        = np.concatenate([cnn_feat.reshape(1,-1), w_scaled], axis=1)
                loss_pct = round(float(model_xgb.predict(X)[0]), 1)

            if loss_pct < 20:
                badge = f'<span class="severity-badge low">🟢 Low Severity — Crop is mostly safe</span>'
            elif loss_pct < 50:
                badge = f'<span class="severity-badge medium">🟡 Moderate Severity — Treatment recommended</span>'
            else:
                badge = f'<span class="severity-badge high">🔴 High Severity — Act immediately!</span>'

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Predicted Yield Loss</div>
                <div class="result-number">{loss_pct}%</div>
                {badge}
                <div style="margin-top:1rem; font-size:0.78rem; color:#475569;">
                    Disease: {selected_disease} · Temp: {temp}°C · Humidity: {humidity}%
                </div>
            </div>
            """, unsafe_allow_html=True)
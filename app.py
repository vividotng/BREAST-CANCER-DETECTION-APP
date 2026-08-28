import streamlit as st
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from model import MammogramResNetV10
import cv2
import numpy as np

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Detection | CNN V10",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PREMIUM MEDICAL VISUAL DESIGN
# ============================================================

st.markdown("""
<style>
/* ---------- GLOBAL ---------- */
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(0, 183, 255, .13), transparent 23%),
        radial-gradient(circle at 92% 20%, rgba(23, 103, 211, .16), transparent 28%),
        linear-gradient(115deg, #010817 0%, #031a36 48%, #010b1d 100%);
    color: #f4f8ff;
}

/* Subtle medical/laboratory atmosphere */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: .075;
    background:
        radial-gradient(circle at 10% 25%, transparent 0 55px, #24b9ff 56px 58px, transparent 59px),
        radial-gradient(circle at 90% 72%, transparent 0 72px, #24b9ff 73px 75px, transparent 76px),
        linear-gradient(125deg, transparent 48%, rgba(31,177,246,.35) 49%, transparent 50%) 8% 82% / 180px 180px no-repeat,
        linear-gradient(35deg, transparent 48%, rgba(31,177,246,.28) 49%, transparent 50%) 86% 15% / 210px 210px no-repeat;
}

/* Decorative medical-equipment silhouettes */
.stApp::after {
    content: "⚕        🔬                         🧬                         🩺";
    position: fixed;
    left: 0;
    right: 0;
    bottom: 18px;
    text-align: center;
    font-size: 34px;
    letter-spacing: 22px;
    color: rgba(43, 190, 255, .055);
    pointer-events: none;
    z-index: 0;
}

.block-container {
    position: relative;
    z-index: 1;
    max-width: 1220px;
    padding-top: 1.25rem;
    padding-bottom: 3rem;
}

/* ---------- HERO ---------- */
.hero-badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 4px 0 14px;
}

.hero-pill {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    border: 1px solid rgba(28, 185, 255, .42);
    background: rgba(3, 38, 76, .66);
    color: #a9e7ff;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .07em;
    text-transform: uppercase;
}

.hero-title {
    text-align: center;
    margin: 5px 0 7px;
    font-size: clamp(2.3rem, 5vw, 4rem);
    line-height: .98;
    font-weight: 900;
    letter-spacing: -.055em;
    color: #f4f8ff;
    text-shadow: 0 8px 30px rgba(0,0,0,.35);
}

.hero-title .blue {
    color: #18c2ff;
}

.hero-subtitle {
    text-align: center;
    color: #aebfd4;
    font-size: 1.18rem;
    margin: 0;
}

.hero-tagline {
    text-align: center;
    color: #56d5ff;
    font-size: .98rem;
    margin: 8px 0 24px;
    letter-spacing: .03em;
}

/* ---------- CARDS ---------- */
.glass-card {
    background: linear-gradient(
        145deg,
        rgba(5, 35, 70, .91),
        rgba(2, 15, 35, .94)
    );
    border: 1px solid rgba(37, 182, 245, .30);
    border-radius: 20px;
    padding: 21px;
    margin-bottom: 16px;
    box-shadow:
        0 20px 55px rgba(0,0,0,.27),
        inset 0 1px 0 rgba(255,255,255,.035);
    backdrop-filter: blur(10px);
}

.card-title {
    color: #17c2ff;
    font-size: .98rem;
    font-weight: 850;
    letter-spacing: .06em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.card-title .num {
    color: #70ddff;
    margin-right: 7px;
}

.about-text {
    color: #c0d1e3;
    line-height: 1.62;
    font-size: .92rem;
}

.feature {
    display: flex;
    gap: 11px;
    margin-top: 15px;
    align-items: flex-start;
}

.feature-icon {
    font-size: 1.35rem;
    width: 30px;
    text-align: center;
}

.feature strong {
    color: #62d8ff;
    display: block;
    margin-bottom: 2px;
}

.feature span {
    color: #9fb4ca;
    font-size: .84rem;
    line-height: 1.45;
}

/* ---------- UPLOAD ---------- */
[data-testid="stFileUploader"] {
    background: rgba(2, 24, 50, .78);
    border: 1px dashed rgba(25, 188, 255, .68);
    border-radius: 17px;
    padding: 9px;
    box-shadow: inset 0 0 32px rgba(0, 142, 220, .065);
}

[data-testid="stFileUploader"] section {
    border: none !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(3, 28, 58, .65) !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #b8d7ee !important;
}

/* ---------- VALIDATION / RESULTS ---------- */
.validation-good {
    background: linear-gradient(135deg, rgba(9, 123, 83, .28), rgba(3, 54, 51, .40));
    border: 1px solid rgba(59, 226, 145, .50);
    border-radius: 16px;
    padding: 18px;
    margin: 12px 0;
}

.validation-bad {
    background: linear-gradient(135deg, rgba(130, 22, 46, .30), rgba(51, 7, 25, .44));
    border: 1px solid rgba(255, 92, 113, .62);
    border-radius: 16px;
    padding: 18px;
    margin: 12px 0;
}

.result-good {
    background: linear-gradient(135deg, rgba(20, 133, 74, .24), rgba(3, 57, 48, .45));
    border: 1px solid rgba(74, 229, 126, .48);
    border-radius: 18px;
    padding: 22px;
}

.result-bad {
    background: linear-gradient(135deg, rgba(135, 27, 52, .25), rgba(46, 8, 27, .46));
    border: 1px solid rgba(255, 99, 121, .50);
    border-radius: 18px;
    padding: 22px;
}

.result-label {
    font-size: 2.15rem;
    font-weight: 900;
    letter-spacing: .02em;
}

.result-good .result-label { color: #54df73; }
.result-bad .result-label { color: #ff7082; }

.probability-card {
    background: rgba(1, 16, 35, .72);
    border: 1px solid rgba(39, 181, 245, .27);
    border-radius: 16px;
    padding: 18px;
}

.note-good,
.note-bad {
    border-radius: 17px;
    padding: 20px 22px;
    margin: 15px 0;
    line-height: 1.65;
}

.note-good {
    background: linear-gradient(135deg, rgba(11, 104, 67, .24), rgba(4, 44, 43, .44));
    border: 1px solid rgba(72, 222, 135, .43);
}

.note-bad {
    background: linear-gradient(135deg, rgba(111, 23, 48, .25), rgba(42, 9, 28, .45));
    border: 1px solid rgba(255, 105, 126, .43);
}

.note-title {
    font-size: 1.2rem;
    font-weight: 800;
    margin-bottom: 7px;
}

.note-good .note-title { color: #62e88b; }
.note-bad .note-title { color: #ff8292; }

.note-text {
    color: #d0deea;
    font-size: .93rem;
}

/* ---------- DISCLAIMER ---------- */
.disclaimer {
    background: rgba(76, 83, 29, .47);
    border: 1px solid rgba(183, 198, 86, .23);
    border-radius: 15px;
    padding: 15px 18px;
    margin: 10px 0 22px;
    color: #f1f2bd;
    line-height: 1.5;
}

/* ---------- IMAGE ---------- */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid rgba(28, 187, 255, .35);
    box-shadow: 0 12px 32px rgba(0,0,0,.24);
}

/* ---------- FOOTER ---------- */
.footer-card {
    text-align: center;
    background: rgba(3, 27, 54, .72);
    border: 1px solid rgba(31, 167, 231, .23);
    border-radius: 18px;
    padding: 18px;
    margin-top: 24px;
}

.footer-main {
    color: #9edfff;
    font-weight: 700;
    font-size: 1rem;
}

.footer-sub {
    color: #718da9;
    font-size: .82rem;
    margin-top: 6px;
}

/* ---------- MOBILE ---------- */
@media (max-width: 760px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero-title {
        font-size: 2.25rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DEVICE + MODEL
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="vividotng/breast-mammogram-cnn-v10",
    filename="mammogram_v10_final.pth"
)

@st.cache_resource
def load_v10_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = MammogramResNetV10(num_classes=2)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero-badges">
    <span class="hero-pill">🧬 Powered by PyTorch • CNN V10</span>
    <span class="hero-pill">💙 AI for Good</span>
</div>

<div class="hero-title">
    BREAST CANCER <span class="blue">DETECTION</span>
</div>

<div class="hero-subtitle">AI-Powered Mammogram Analysis</div>
<div class="hero-tagline">Early Detection &nbsp;·&nbsp; Better Outcomes &nbsp;·&nbsp; Stronger Hope</div>
""", unsafe_allow_html=True)

# ============================================================
# ABOUT + DISCLAIMER
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="card-title">ABOUT THIS APP</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="about-text">This educational application uses deep learning with PyTorch and CNN V10 to classify mammogram images as benign or malignant.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p style="color:#62d8ff;font-weight:800;margin-top:18px;">🛡️ Secure &amp; Private</p>'
        '<p style="color:#9fb4ca;font-size:0.84rem;">Images are processed by the application and are not intended for storage.</p>'
        '<p style="color:#62d8ff;font-weight:800;margin-top:15px;">🧠 Deep Learning</p>'
        '<p style="color:#9fb4ca;font-size:0.84rem;">CNN V10 trained for mammogram classification.</p>'
        '<p style="color:#62d8ff;font-weight:800;margin-top:15px;">🎯 Two-Stage Analysis</p>'
        '<p style="color:#9fb4ca;font-size:0.84rem;">The application first checks whether the uploaded image appears to be a mammogram before running the cancer classification model.</p>'
        '<p style="color:#62d8ff;font-weight:800;margin-top:15px;">❤️ For Support</p>'
        '<p style="color:#9fb4ca;font-size:0.84rem;">AI output is educational and should always be discussed with a qualified healthcare professional.</p>',
        unsafe_allow_html=True
    )

uploaded = st.file_uploader(
        "Upload a mammogram JPG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

# ============================================================
# AUTOMATIC MAMMOGRAM VALIDITY CHECK
# ============================================================

def looks_like_mammogram(image):
    """
    Automatic mammogram input gate.
    This is an image-type gate only; it does NOT diagnose cancer.
    """
    gray = np.array(image.convert("L"))

    if gray.size == 0:
        return False

    img = cv2.resize(gray, (512, 512))

    dark_ratio = np.mean(img < 30)
    bright_ratio = np.mean(img > 220)
    contrast = np.std(img)

    _, thresh = cv2.threshold(
        img,
        15,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return False

    largest = max(contours, key=cv2.contourArea)

    area_ratio = cv2.contourArea(largest) / (512 * 512)

    if (
        dark_ratio > 0.30
        and contrast > 20
        and bright_ratio < 0.15
        and area_ratio > 0.20
    ):
        return True

    return False

# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded:

    try:
        image = Image.open(uploaded).convert("L")
    except Exception:
        st.error("The uploaded file could not be read as an image.")
        st.stop()

    st.markdown("""
    <div class="card-title" style="margin-top:18px;">
        <span class="num">2.</span> Mammogram Validation
    </div>
    """, unsafe_allow_html=True)

    img_col, info_col = st.columns([1.05, 1], gap="large")

    with img_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # STAGE 1 — IMAGE TYPE GATE
    # --------------------------------------------------------

    if not looks_like_mammogram(image):

        with info_col:
            st.markdown("""
            <div class="validation-bad">
                <div style="font-size:1.2rem;font-weight:800;color:#ff7b8c;">
                    ⚠️ IMAGE NOT ACCEPTED
                </div>
                <div style="color:#e0c5cb;margin-top:7px;line-height:1.55;">
                    The uploaded image does not appear to be a breast mammogram.
                    Please upload a valid mammogram image for analysis.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="glass-card">
                <div class="card-title">Accepted Input</div>
                <div class="about-text">
                    The application is designed for breast mammogram images.
                    Other images such as people, animals, ordinary photographs,
                    chest X-rays, brain scans, objects or scenery should not be
                    sent to the cancer-classification model.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.stop()

    with info_col:
        st.markdown("""
        <div class="validation-good">
            <div style="font-size:1.2rem;font-weight:800;color:#58e38b;">
                ✓ VALID MAMMOGRAM DETECTED
            </div>
            <div style="color:#c3e5d5;margin-top:7px;line-height:1.55;">
                The uploaded image passed the mammogram input check.
                Proceeding to CNN V10 classification.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # STAGE 2 — V10 CLASSIFICATION
    # ========================================================

    if not Path(MODEL_PATH).exists():
        st.error(f"Trained V10 model not found: {MODEL_PATH}")
        st.stop()

    try:
        model = load_v10_model()
    except Exception as e:
        st.error(f"Unable to load the V10 model: {e}")
        st.stop()

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    benign = float(probs[0])
    malignant = float(probs[1])

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("""
    <div class="card-title" style="margin-top:22px;">
        <span class="num">3.</span> Analysis Result — CNN V10
    </div>
    """, unsafe_allow_html=True)

    result_left, result_right = st.columns([1.05, 1], gap="large")

    if malignant >= benign:

        with result_left:
            st.markdown(f"""
            <div class="result-bad">
                <div style="color:#ff9aaa;font-size:.84rem;font-weight:800;letter-spacing:.08em;">
                    MODEL CLASSIFICATION
                </div>
                <div class="result-label">MALIGNANT</div>
                <div style="color:#d8e4ee;margin-top:7px;">
                    Malignant probability: <strong>{malignant:.1%}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        note_class = "note-bad"
        note_title = "❤️ Further Medical Evaluation Recommended"
        note_text = (
            "The model identified image patterns that it associated with malignancy. "
            "<strong>This AI prediction does not by itself confirm breast cancer.</strong> "
            "Please do not lose hope. Further diagnostic tests and clinical evaluation "
            "are important to confirm the finding and determine the appropriate next steps. "
            "If a healthcare professional recommends additional imaging, biopsy or other "
            "tests, following that advice is important."
        )

    else:

        with result_left:
            st.markdown(f"""
            <div class="result-good">
                <div style="color:#8ce5aa;font-size:.84rem;font-weight:800;letter-spacing:.08em;">
                    MODEL CLASSIFICATION
                </div>
                <div class="result-label">BENIGN</div>
                <div style="color:#d8e4ee;margin-top:7px;">
                    Benign probability: <strong>{benign:.1%}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        note_class = "note-good"
        note_title = "🎉 Reassuring Result"
        note_text = (
            "The model classified this mammogram as <strong>BENIGN</strong>. "
            "This means the image did not show patterns that the model associated "
            "with malignancy. Continue taking care of your health through healthy "
            "lifestyle choices, regular medical check-ups and recommended breast "
            "screening. Remember that an AI prediction is not a medical diagnosis."
        )

    with result_right:
      st.markdown(
        '<div class="card-title">PREDICTION PROBABILITIES</div>',
        unsafe_allow_html=True
    )

    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        st.metric(
            label="Benign",
            value=f"{benign:.1%}"
        )

    with prob_col2:
        st.metric(
            label="Malignant",
            value=f"{malignant:.1%}"
        )

    st.write("Benign")
    st.progress(benign)

    st.write("Malignant")
    st.progress(malignant)

    st.caption(
        "Probabilities represent the model's output for the two classes."
    )
    # ========================================================
    # SUPPORTIVE EDUCATIONAL NOTE
    # ========================================================

    st.markdown("""
    <div class="card-title" style="margin-top:22px;">
        <span class="num">4.</span> Important Message
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="{note_class}">
        <div class="note-title">{note_title}</div>
        <div class="note-text">{note_text}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer-card">
    <div class="footer-main">💙 Your health matters. Stay informed. Stay proactive.</div>
    <div class="footer-sub">
        AI for Good • Educational CBIS-DDSM Research Project • CNN V10
        <br>
        This application does not replace professional medical advice, diagnosis or treatment.
    </div>
</div>
""", unsafe_allow_html=True)

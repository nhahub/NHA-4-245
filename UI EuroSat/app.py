import numpy as np
import pandas as pd
import streamlit as st
import tifffile as tiff
from skimage.transform import resize
from tensorflow.keras.models import load_model

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = (64, 64)

MODEL_REGISTRY = {
    "EfficientNet": {
        "file": "EfficientNet_final.h5",
        "icon": "🧠",
        "type": "Transfer learning",
        "size_mb": 50.7,
        "group": "transfer",
    },
    "ResNet50": {
        "file": "ResNet50_final.h5",
        "icon": "🏛️",
        "type": "Transfer learning",
        "size_mb": 276.7,
        "group": "transfer",
    },
    "SimpleCNN": {
        "file": "SimpleCNN_final.h5",
        "icon": "⚡",
        "type": "Custom CNN",
        "size_mb": 54.3,
        "group": "lightweight",
    },
    "SpectrumNet": {
        "file": "SpectrumNet_final.h5",
        "icon": "📶",
        "type": "Custom spectral CNN",
        "size_mb": 1.0,
        "group": "lightweight",
    },
}
GROUP_LABELS = {
    "transfer": "Transfer-learning models",
    "lightweight": "Lightweight & custom models",
}
GROUP_COLORS = {
    "transfer": "#3b82f6",     # blue
    "lightweight": "#22c55e",  # green
}

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
CLASS_DESCRIPTIONS = {
    "AnnualCrop": "Land used for crops replanted every season",
    "Forest": "Tree-covered land",
    "HerbaceousVegetation": "Natural grass and non-woody vegetation",
    "Highway": "Roads and highway infrastructure",
    "Industrial": "Factories, warehouses, industrial zones",
    "Pasture": "Grazing land for livestock",
    "PermanentCrop": "Orchards, vineyards, and similar long-term crops",
    "Residential": "Housing and urban residential areas",
    "River": "Rivers and flowing water bodies",
    "SeaLake": "Seas, lakes, and large standing water bodies",
}

st.set_page_config(
    page_title="Land Cover Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0a0f1e;
        }
        .main .block-container { padding-top: 2rem; max-width: 1200px; }

        .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
            color: #e2e8f0;
        }
        .stApp .stCaption, .stApp small { color: #94a3b8 !important; }

        h1 { font-weight: 700; letter-spacing: -0.02em; }
        h2, h3 { font-weight: 600; }

        .app-subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-top: -0.6rem;
            margin-bottom: 1.8rem;
        }

        .result-card {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            border-radius: 14px;
            padding: 1.6rem 1.8rem;
            color: white;
            margin-bottom: 1rem;
        }
        .result-card .label { font-size: 0.85rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.06em; }
        .result-card .value { font-size: 2rem; font-weight: 700; margin-top: 0.2rem; }
        .result-card .desc { opacity: 0.75; margin-top: 0.4rem; font-size: 0.95rem; }

        .info-pill {
            display: inline-block;
            background: #f1f5f9;
            border-radius: 999px;
            padding: 0.15rem 0.7rem;
            font-size: 0.82rem;
            color: #334155;
            margin-right: 0.4rem;
        }

        section[data-testid="stSidebar"] {
            background-color: #0f172a;
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.55rem 1.4rem;
        }

        /* --- Model picker (LandViewer style) --- */
        .group-pill {
            display: inline-block;
            border: 1.5px solid;
            border-radius: 999px;
            padding: 0.5rem 1.4rem;
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 1rem;
        }

        .model-card {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            background: #111827;
            border: 1.5px solid #1f2937;
            border-radius: 14px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.7rem;
            transition: border-color 0.15s ease;
        }
        .model-card.selected {
            border-color: var(--accent, #3b82f6);
            background: #131c2e;
        }
        .model-card .icon-circle {
            width: 46px;
            height: 46px;
            min-width: 46px;
            border-radius: 50%;
            border: 1.5px solid var(--accent, #3b82f6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            background: #0b1220;
        }
        .model-card .model-name {
            font-weight: 700;
            color: #f8fafc;
            font-size: 1rem;
        }
        .model-card .model-detail {
            color: #94a3b8;
            font-size: 0.85rem;
            line-height: 1.35;
        }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Model loading (cached per model file, so switching is fast after first load)
# ----------------------------
@st.cache_resource
def get_model(model_path):
    return load_model(model_path)


# ----------------------------
# Preprocessing (same logic as your notebook)
# ----------------------------
RED_BAND = 3
NIR_BAND = 7
NDVI_EPSILON = 1e-6


def add_ndvi_channel(img):
    """Compute NDVI from the Red and NIR bands and append it as a 14th channel,
    matching the training notebook's preprocessing."""
    red = img[..., RED_BAND]
    nir = img[..., NIR_BAND]
    ndvi = (nir - red) / (nir + red + NDVI_EPSILON)
    return np.concatenate([img, ndvi[..., np.newaxis]], axis=-1)


def preprocess_image(uploaded_file):
    img = tiff.imread(uploaded_file)

    if img.ndim != 3 or img.shape[-1] != 13:
        raise ValueError(
            f"Expected a 13-band image, but got shape {img.shape}."
        )
    img = img.astype(np.float32)

    resized = resize(
        img,
        (IMG_SIZE[0], IMG_SIZE[1], 13),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    # Models are trained on 13 raw bands + an NDVI channel (14 channels total)
    model_input = add_ndvi_channel(resized)
    model_input = np.expand_dims(model_input, axis=0)
    return img, model_input


def make_rgb_preview(img):
    """Build a viewable RGB image from Sentinel-2 style 13-band data (bands 4,3,2 = R,G,B)."""
    r, g, b = img[..., 3], img[..., 2], img[..., 1]
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return rgb


def make_globe_svg():
    """A decorative glowing wireframe-globe graphic, styled after a network/orbit motif."""
    import random

    random.seed(7)
    cx, cy, r = 150, 150, 120

    # Latitude ellipses (horizontal rings, squashed vertically)
    lat_lines = "".join(
        f'<ellipse cx="{cx}" cy="{cy}" rx="{r}" ry="{ry}" '
        f'fill="none" stroke="#5eead4" stroke-opacity="0.35" stroke-width="1"/>'
        for ry in [20, 45, 70, 95]
    )
    # Longitude ellipses (vertical rings, squashed horizontally)
    lon_lines = "".join(
        f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{r}" '
        f'fill="none" stroke="#5eead4" stroke-opacity="0.35" stroke-width="1"/>'
        for rx in [20, 45, 70, 95]
    )

    # Scattered network nodes (dots) around the sphere surface
    nodes = []
    for _ in range(16):
        ang = random.uniform(0, 6.283)
        rad = random.uniform(0.3, 1.0) * r
        x = cx + rad * np.cos(ang)
        y = cy + rad * np.sin(ang) * 0.55  # flatten for sphere illusion
        nodes.append((x, y))

    dots_svg = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="#a7f3d0"/>' for x, y in nodes
    )

    # A few connecting lines between nearby nodes for the "network" look
    lines_svg = ""
    for i in range(0, len(nodes) - 1, 2):
        x1, y1 = nodes[i]
        x2, y2 = nodes[i + 1]
        lines_svg += (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#5eead4" stroke-opacity="0.4" stroke-width="1"/>'
        )

    return f"""
    <svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:300px;">
        <defs>
            <radialGradient id="glow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#134e4a" stop-opacity="0.9"/>
                <stop offset="70%" stop-color="#0f766e" stop-opacity="0.35"/>
                <stop offset="100%" stop-color="#0f766e" stop-opacity="0"/>
            </radialGradient>
            <filter id="softGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
        <circle cx="{cx}" cy="{cy}" r="{r + 25}" fill="url(#glow)"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="#0b2b28" fill-opacity="0.6"
                stroke="#2dd4bf" stroke-opacity="0.5" stroke-width="1.5"/>
        {lat_lines}
        {lon_lines}
        <g filter="url(#softGlow)">
            {lines_svg}
            {dots_svg}
        </g>
    </svg>
    """


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### 🛰️ Model Info")
    st.markdown(
        """
        <span class="info-pill" style="background:#1e293b;color:#e2e8f0;">4 models available</span>
        <span class="info-pill" style="background:#1e293b;color:#e2e8f0;">13-band upload</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("**Input requirements**")
    st.write(
        "- Format: `.tif` / `.tiff` only\n"
        "- Bands: 13 (Sentinel-2 style)\n"
        "- Resized internally to 64×64\n"
        "- NDVI is computed and added as a 14th channel before prediction"
    )

    st.write("")
    st.markdown("**Classes recognized**")
    for c in CLASSES:
        st.markdown(f"- {c}")

    st.write("")
    st.caption("Model files expected in the app directory, alongside app.py.")

# ----------------------------
# Header
# ----------------------------
st.markdown(
    '<h1 style="text-align:center;">Available Land Cover Models</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle" style="text-align:center;">'
    'Upload a multispectral satellite image, pick a model below, and classify its land cover type.'
    '</div>',
    unsafe_allow_html=True,
)

# ----------------------------
# Model picker (hero layout: cards left/right, globe centered)
# ----------------------------
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "EfficientNet"

pill_l, _, pill_r = st.columns([1, 0.4, 1])
with pill_l:
    st.markdown(
        f'<div style="text-align:center;">'
        f'<span class="group-pill" style="border-color:{GROUP_COLORS["transfer"]};'
        f'color:{GROUP_COLORS["transfer"]};">{GROUP_LABELS["transfer"]}</span></div>',
        unsafe_allow_html=True,
    )
with pill_r:
    st.markdown(
        f'<div style="text-align:center;">'
        f'<span class="group-pill" style="border-color:{GROUP_COLORS["lightweight"]};'
        f'color:{GROUP_COLORS["lightweight"]};">{GROUP_LABELS["lightweight"]}</span></div>',
        unsafe_allow_html=True,
    )

left_col, center_col, right_col = st.columns([1, 0.9, 1], gap="medium")
column_for_group = {"transfer": left_col, "lightweight": right_col}

with center_col:
    st.markdown(
        f'<div style="display:flex;justify-content:center;align-items:center;'
        f'padding-top:1.5rem;">{make_globe_svg()}</div>',
        unsafe_allow_html=True,
    )

for name, info in MODEL_REGISTRY.items():
    color = GROUP_COLORS[info["group"]]
    col = column_for_group[info["group"]]
    with col:
        is_selected = st.session_state.selected_model == name
        card_class = "model-card selected" if is_selected else "model-card"
        st.markdown(
            f"""
            <div class="{card_class}" style="--accent:{color};">
                <div class="icon-circle" style="border-color:{color};">{info['icon']}</div>
                <div>
                    <div class="model-name">{name}</div>
                    <div class="model-detail">{info['type']}<br>Size: {info['size_mb']:.1f} MB</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            f"{'✓ Selected' if is_selected else 'Select'}",
            key=f"select_{name}",
            use_container_width=True,
            disabled=is_selected,
        ):
            st.session_state.selected_model = name
            st.rerun()

selected_info = MODEL_REGISTRY[st.session_state.selected_model]
MODEL_PATH = selected_info["file"]
st.markdown(
    f'<div style="text-align:center;">Using <b>{st.session_state.selected_model}</b> '
    f'for prediction (<code>{MODEL_PATH}</code>)</div>',
    unsafe_allow_html=True,
)
st.write("")
st.divider()



# ----------------------------
# Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a .tif image",
    type=["tif", "tiff"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("📤 Upload a 13-band `.tif` satellite image to get started.")
    st.stop()

try:
    raw_img, model_input = preprocess_image(uploaded_file)
except Exception as e:
    st.error(
        "⚠️ This doesn't look like a valid 13-band satellite `.tif` image "
        f"({e}). Please convert your image to a 13-band `.tif` file and "
        "upload that instead."
    )
    st.stop()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Preview")
    st.image(make_rgb_preview(raw_img), use_container_width=True)
    st.caption(
        f"Shape: {raw_img.shape}  •  Bands: {raw_img.shape[-1]}  •  Dtype: {raw_img.dtype}"
    )

with right:
    st.subheader("Classification")
    run = st.button("Run prediction", type="primary", use_container_width=True)

    if run:
        with st.spinner(f"Loading {st.session_state.selected_model} and running inference..."):
            try:
                model = get_model(MODEL_PATH)
            except Exception as e:
                st.error(
                    f"Couldn't load '{MODEL_PATH}'. Make sure the model file is "
                    f"in the same folder as this app. Error: {e}"
                )
                st.stop()

            prediction = model.predict(model_input, verbose=0)[0]
            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[predicted_class])

        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">Predicted class</div>
                <div class="value">{CLASSES[predicted_class]}</div>
                <div class="desc">{CLASS_DESCRIPTIONS[CLASSES[predicted_class]]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        m1.metric("Confidence", f"{confidence:.1%}")
        m2.metric("Second most likely", CLASSES[int(np.argsort(prediction)[-2])])

        st.write("")
        st.markdown("**Class probabilities**")
        df = pd.DataFrame(
            {"Class": CLASSES, "Probability (%)": prediction * 100}
        ).sort_values("Probability (%)", ascending=False).reset_index(drop=True)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Probability (%)": st.column_config.ProgressColumn(
                    "Probability", format="%.1f%%", min_value=0, max_value=100
                )
            },
        )
    else:
        st.caption("Click **Run prediction** to classify this image.")

st.write("")
st.divider()
st.caption("Built with Streamlit · Land-cover classification with EfficientNet, ResNet50, SimpleCNN & SpectrumNet")
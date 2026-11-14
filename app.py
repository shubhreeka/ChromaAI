import base64
import io
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image

from src.architecture_tab import render_architecture_and_why
from src.infer import ROOT, available_models, best_info, compose, load_model, timed_predict
from src.models import DISPLAY_NAMES

st.set_page_config(page_title="ChromaAI", page_icon="🎨", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSS = """
<style>
.stApp{background:radial-gradient(1100px 600px at 8% -10%,#2b1a52 0%,transparent 60%),radial-gradient(900px 500px at 100% 0%,#0e3b4c 0%,transparent 55%),#0a0c13;color:#e9ebf3}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2rem;max-width:1200px}
.hero h1{font-size:3.6rem;font-weight:800;line-height:1.05;margin:0 0 .4rem 0;background:linear-gradient(90deg,#ff7a59,#ffcf5c,#4de1c1,#6a8cff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:#a0a9c0;font-size:1.15rem;max-width:760px;margin:0 0 1.2rem 0}
.chips{display:flex;gap:.6rem;flex-wrap:wrap;margin:.4rem 0 1.2rem 0}
.chip{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:999px;padding:.35rem .9rem;font-size:.85rem;color:#cfd5e6}
.chip b{color:#fff}
.card{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.09);border-radius:18px;padding:1.2rem 1.4rem;backdrop-filter:blur(10px)}
.card h4{margin:0 0 .4rem 0;color:#fff}
.card p{margin:0;color:#a8b0c5}
.best{background:linear-gradient(135deg,rgba(77,225,193,.18),rgba(106,140,255,.18));border:1px solid rgba(77,225,193,.4);border-radius:18px;padding:1.1rem 1.4rem;margin-bottom:1rem}
.best h3{margin:0;color:#fff}
.best p{margin:.2rem 0 0 0;color:#b9c2d8}
[data-testid="stSidebar"]{background:#0d101a;border-right:1px solid rgba(255,255,255,.06)}
[data-testid="stFileUploader"] section{background:rgba(255,255,255,.04);border:2px dashed rgba(106,140,255,.45);border-radius:18px;padding:1.6rem}
.stButton button,.stDownloadButton button{background:linear-gradient(90deg,#6a8cff,#4de1c1);color:#0a0c13;border:0;border-radius:12px;font-weight:700;padding:.65rem 1.5rem}
.stButton button:hover,.stDownloadButton button:hover{filter:brightness(1.1);color:#0a0c13}
.stTabs [data-baseweb="tab-list"]{gap:.5rem}
.stTabs [data-baseweb="tab"]{background:rgba(255,255,255,.04);border-radius:12px;padding:.5rem 1.2rem}
.stTabs [aria-selected="true"]{background:rgba(106,140,255,.25)}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model(name, size):
    return load_model(name, size, DEVICE)


@st.cache_data(show_spinner=False)
def run_model(data, name, size, max_side, _model):
    image = Image.open(io.BytesIO(data))
    return timed_predict(image, _model, size, DEVICE, max_side)


def to_b64(img, max_side=1100):
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def slider_html(gray_b64, color_b64):
    return f"""
<style>
html,body{{margin:0;background:transparent;font-family:system-ui,sans-serif}}
.wrap{{position:relative;width:100%;height:620px;border-radius:18px;overflow:hidden;background:#05060a;border:1px solid rgba(255,255,255,.12)}}
.wrap img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;user-select:none;pointer-events:none}}
#top{{clip-path:inset(0 50% 0 0)}}
.line{{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;box-shadow:0 0 12px rgba(0,0,0,.7);pointer-events:none}}
.knob{{position:absolute;top:50%;left:50%;width:44px;height:44px;margin:-22px 0 0 -22px;border-radius:50%;background:#fff;color:#111;display:flex;align-items:center;justify-content:center;font-weight:800;box-shadow:0 4px 16px rgba(0,0,0,.5);pointer-events:none}}
.tag{{position:absolute;top:14px;padding:6px 14px;border-radius:999px;background:rgba(0,0,0,.6);color:#fff;font-size:13px;pointer-events:none}}
input{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}}
</style>
<div class="wrap">
<img src="data:image/jpeg;base64,{color_b64}">
<img id="top" src="data:image/jpeg;base64,{gray_b64}">
<div class="line" id="line"></div>
<div class="knob" id="knob">&harr;</div>
<span class="tag" style="left:14px">Original</span>
<span class="tag" style="right:14px">Colorized</span>
<input type="range" min="0" max="100" value="50" id="r">
</div>
<script>
const r=document.getElementById('r'),t=document.getElementById('top'),l=document.getElementById('line'),k=document.getElementById('knob');
r.addEventListener('input',()=>{{const v=r.value;t.style.clipPath='inset(0 '+(100-v)+'% 0 0)';l.style.left=v+'%';k.style.left=v+'%';}});
</script>
"""


st.markdown(
    """
<div class="hero">
<h1>ChromaAI</h1>
<p>Bring black-and-white photographs back to life. Upload a grayscale image and a deep U-Net model, selected from a five-model comparison, predicts its colors in seconds.</p>
</div>
""",
    unsafe_allow_html=True,
)

info = best_info()
models = available_models()

if info is None or not models:
    st.error("No trained model found. Run the comparison first:  python -m src.compare")
    st.stop()

with st.sidebar:
    st.markdown("### Settings")
    default_idx = models.index(info["name"]) if info["name"] in models else 0
    model_name = st.selectbox(
        "Model",
        models,
        index=default_idx,
        format_func=lambda n: DISPLAY_NAMES[n] + ("  (best)" if n == info["name"] else ""),
    )
    vibrance = st.slider("Color vibrance", 0.5, 2.5, 1.3, 0.05, help="Boosts color saturation. L1-trained models tend to produce muted colors.")
    max_side = st.select_slider("Max output size (px)", options=[512, 768, 1024, 1536], value=1024)
    st.markdown("---")
    st.caption(f"Device: {'GPU' if DEVICE.type == 'cuda' else 'CPU'}")
    st.caption(f"Model input size: {info['image_size']} px")

model = get_model(model_name, info["image_size"])

tab_colorize, tab_compare, tab_arch, tab_about = st.tabs(
    ["🎨  Colorize", "📊  Model comparison", "🏗️  Architecture & Why U-Net", "ℹ️  How it works"]
)

with tab_colorize:
    uploaded = st.file_uploader("Drop a grayscale image here", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if uploaded is None:
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="card"><h4>1 · Upload</h4><p>Any black-and-white or color photo. Color images are converted to grayscale first.</p></div>', unsafe_allow_html=True)
        c2.markdown('<div class="card"><h4>2 · Predict</h4><p>The model reads the lightness channel and predicts the a and b color channels.</p></div>', unsafe_allow_html=True)
        c3.markdown('<div class="card"><h4>3 · Compare</h4><p>Drag the slider to compare before and after, tune vibrance, then download.</p></div>', unsafe_allow_html=True)
    else:
        data = uploaded.getvalue()
        with st.spinner("Colorizing..."):
            L_full, ab_full, seconds = run_model(data, model_name, info["image_size"], max_side, model)
        gray_img = compose(L_full, ab_full, 0.0)
        color_img = compose(L_full, ab_full, vibrance)

        st.markdown(
            f'<div class="chips"><span class="chip">Model <b>{DISPLAY_NAMES[model_name]}</b></span>'
            f'<span class="chip">Size <b>{color_img.width}×{color_img.height}</b></span>'
            f'<span class="chip">Inference <b>{seconds:.2f}s</b></span>'
            f'<span class="chip">Vibrance <b>{vibrance:.2f}×</b></span></div>',
            unsafe_allow_html=True,
        )

        view = st.radio("View", ["Slider", "Side by side"], horizontal=True, label_visibility="collapsed")
        if view == "Slider":
            components.html(slider_html(to_b64(gray_img), to_b64(color_img)), height=640)
        else:
            a, b = st.columns(2)
            a.image(gray_img, caption="Original", width="stretch")
            b.image(color_img, caption="Colorized", width="stretch")

        buf = io.BytesIO()
        color_img.save(buf, format="PNG")
        st.download_button("⬇  Download colorized image", buf.getvalue(), file_name="colorized.png", mime="image/png")

with tab_compare:
    csv = ROOT / "results" / "comparison.csv"
    if not csv.exists():
        st.info("Comparison results not found. Run:  python -m src.compare")
    else:
        df = pd.read_csv(csv)
        top = df.iloc[0]
        st.markdown(
            f'<div class="best"><h3>🏆 Selected model: {top["display_name"]}</h3>'
            f'<p>PSNR {top["psnr"]:.2f} dB · SSIM {top["ssim"]:.3f} · L1 {top["test_l1"]:.4f} · {top["params_m"]:.1f}M parameters</p></div>',
            unsafe_allow_html=True,
        )
        view_df = df[["display_name", "params_m", "train_minutes", "best_val_loss", "test_l1", "psnr", "ssim", "avg_rank"]].rename(
            columns={
                "display_name": "Model",
                "params_m": "Params (M)",
                "train_minutes": "Train (min)",
                "best_val_loss": "Val L1",
                "test_l1": "Test L1",
                "psnr": "PSNR",
                "ssim": "SSIM",
                "avg_rank": "Avg rank",
            }
        )
        st.dataframe(view_df, width="stretch", hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown("**PSNR** (higher is better)")
        c1.bar_chart(df.set_index("display_name")["psnr"])
        c2.markdown("**SSIM** (higher is better)")
        c2.bar_chart(df.set_index("display_name")["ssim"])
        c3.markdown("**Test L1** (lower is better)")
        c3.bar_chart(df.set_index("display_name")["test_l1"])
        full = ROOT / "results" / "full_comparison.csv"
        if full.exists():
            st.markdown("### Color-family accuracy and colorfulness")
            fdf = pd.read_csv(full)
            st.dataframe(
                fdf[["display_name", "accuracy", "balanced_accuracy", "pred_chroma", "true_chroma", "psnr", "ssim"]].rename(
                    columns={
                        "display_name": "Model",
                        "accuracy": "Pixel accuracy",
                        "balanced_accuracy": "Balanced accuracy",
                        "pred_chroma": "Predicted chroma",
                        "true_chroma": "True chroma",
                        "psnr": "PSNR",
                        "ssim": "SSIM",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            cm_img = ROOT / "results" / "confusion_matrices.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion matrices (rows: true color family, columns: predicted)", width="stretch")
        curve = ROOT / "results" / "loss_curves.png"
        if curve.exists():
            st.image(str(curve), caption="Validation loss per epoch", width="stretch")

with tab_arch:
    render_architecture_and_why()

with tab_about:
    st.markdown(
        """
<div class="card">
<h4>Pipeline</h4>
<p>Image → resize → convert to Lab → the lightness channel (L) is the grayscale input → the network predicts the color channels (a, b) → predicted ab is upsampled to the original resolution and merged with the original L → converted back to RGB. Because the original lightness is kept at full resolution, the output retains all of the source image's detail.</p>
</div>
<br>
<div class="card">
<h4>Model selection</h4>
<p>Five architectures (Conv Autoencoder, Classic U-Net, Attention U-Net, Pix2Pix U-Net, ResNet18 U-Net) are trained under identical conditions on a COCO subset and evaluated on a held-out test split using PSNR, SSIM, and L1 error. The model with the best average rank across the three metrics is used as the default.</p>
</div>
""",
        unsafe_allow_html=True,
    )
"""
src/architecture_tab.py

Adds two tabs to the Streamlit app:
    render_architecture_tab()  -> animated architecture of every model, how they relate, best model
    render_why_unet_tab()      -> why U-Net, with ready answers for viva questions

Nothing here trains or changes your models. Live parameter counts and layer
listings are read from your real code when possible; if anything fails the tab
falls back to the static text, so the app never breaks.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------
MODELS = [
    {
        "key": "autoencoder",
        "title": "1. Convolutional Autoencoder (baseline)",
        "enc": ["Conv 64", "Conv 128", "Conv 256", "Bottleneck"],
        "dec": ["Up 256", "Up 128", "Up 64"],
        "skips": False, "gate": False, "pretrained": False,
        "idea": "Compresses the L (grayscale) image into a small code, then rebuilds the a and b color channels from that code.",
        "adds": "Starting point. No shortcuts between encoder and decoder.",
        "weak": "Fine spatial detail is squeezed through the bottleneck, so edges and textures can blur.",
        "loss": "L1 loss on the a/b channels",
    },
    {
        "key": "classic_unet",
        "title": "2. Classic U-Net",
        "enc": ["Conv 64", "Conv 128", "Conv 256", "Bottleneck"],
        "dec": ["Up 256", "Up 128", "Up 64"],
        "skips": True, "gate": False, "pretrained": False,
        "idea": "Same encoder-decoder shape, plus skip connections that copy encoder features straight to the matching decoder level.",
        "adds": "Skip connections: edges and textures from L reach the output without being compressed away.",
        "weak": "Every skip is passed on unfiltered, including features that are not useful.",
        "loss": "L1 loss on the a/b channels",
    },
    {
        "key": "attention_unet",
        "title": "3. Attention U-Net",
        "enc": ["Conv 64", "Conv 128", "Conv 256", "Bottleneck"],
        "dec": ["Up 256", "Up 128", "Up 64"],
        "skips": True, "gate": True, "pretrained": False,
        "idea": "A U-Net where each skip passes through an attention gate that decides which encoder features are relevant before they are merged.",
        "adds": "Attention gates on the skip connections.",
        "weak": "More parameters and more complexity; on our small dataset it performed about the same as the classic U-Net.",
        "loss": "L1 loss on the a/b channels",
    },
    {
        "key": "pix2pix_unet",
        "title": "4. Pix2Pix-style U-Net (large)",
        "enc": ["Down 64", "Down 128", "Down 256", "Down 512", "Bottleneck"],
        "dec": ["Up 512", "Up 256", "Up 128", "Up 64"],
        "skips": True, "gate": False, "pretrained": False,
        "idea": "A much deeper and wider U-Net using the generator design from the pix2pix paper (about 41.8M parameters). Only the U-Net generator is used. There is no discriminator, so this is not a GAN.",
        "adds": "Depth and width: many more layers and channels.",
        "weak": "Too big for 3,500 images and 8 epochs, so it finished last. Bigger is not automatically better.",
        "loss": "L1 loss on the a/b channels",
    },
    {
        "key": "resnet18_unet",
        "title": "5. ResNet18 U-Net (best model)",
        "enc": ["ResNet18 stem", "Layer 1", "Layer 2", "Layer 3", "Layer 4"],
        "dec": ["Up", "Up", "Up", "Up"],
        "skips": True, "gate": False, "pretrained": True,
        "idea": "A U-Net whose encoder is ResNet18 pretrained on ImageNet. It already understands objects and textures before it sees our data.",
        "adds": "A pretrained encoder (transfer learning).",
        "weak": "Still trained with L1 loss, so colors tend to be muted when the model is unsure.",
        "loss": "L1 loss on the a/b channels",
    },
    {
        "key": "resnet18_class",
        "title": "6. ResNet18 U-Net with color classification",
        "enc": ["ResNet18 stem", "Layer 1", "Layer 2", "Layer 3", "Layer 4"],
        "dec": ["Up", "Up", "Up", "Up"],
        "skips": True, "gate": False, "pretrained": True,
        "classes": True,
        "idea": "The same ResNet18 U-Net, but each pixel chooses one of 484 color bins (a 22x22 grid over a/b) instead of regressing a value. Rare, vivid colors get higher loss weight.",
        "adds": "Classification head and class-weighted cross-entropy (after Zhang et al., 2016) to fight the muted-color problem.",
        "weak": "Pixel metrics (PSNR, SSIM) are lower because they reward safe averages. Judge it by eye and by chroma.",
        "loss": "Class-weighted cross-entropy over 484 bins",
    },
]

BEST_MODEL_KEY = "resnet18_unet"
BEST_NUMBERS = {"PSNR": "24.29 dB", "SSIM": "0.924", "L1 error": "0.0735"}


# --------------------------------------------------------------------------
# Animated diagram (pure SVG + CSS, no extra libraries)
# --------------------------------------------------------------------------
def _diagram_html(m):
    enc, dec = m["enc"], m["dec"]
    n = len(enc)
    total_cols = 2 * n + 1
    W, H = 900, 340
    col_w = W / total_cols
    cy = 175

    def block_h(level):
        return 170 - level * (110 / max(n - 1, 1))

    parts = []
    centers = []  # (x_center, height, label, color)

    def add(col, h, label, color):
        x = col * col_w + col_w * 0.12
        w = col_w * 0.76
        centers.append((x + w / 2, h))
        parts.append(
            f'<rect x="{x:.1f}" y="{cy - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" '
            f'fill="{color}" class="blk"/>'
            f'<text x="{x + w / 2:.1f}" y="{cy + h / 2 + 18:.1f}" text-anchor="middle" class="lbl">{label}</text>'
        )

    enc_color = "#7c5cff" if m.get("pretrained") else "#3b82f6"
    dec_color = "#10b981"
    add(0, 170, "L input", "#94a3b8")
    for i, lab in enumerate(enc):
        add(1 + i, block_h(i), lab, enc_color)
    for j, lab in enumerate(dec):
        add(n + 1 + j, block_h(n - 2 - j), lab, dec_color)
    out_label = "484-bin map" if m.get("classes") else "a, b output"
    add(2 * n, 170, out_label, "#f59e0b")

    lines = []
    # forward flow arrows between neighbouring blocks
    for i in range(len(centers) - 1):
        x1, h1 = centers[i]
        x2, h2 = centers[i + 1]
        lines.append(
            f'<line x1="{x1 + col_w * 0.38:.1f}" y1="{cy}" x2="{x2 - col_w * 0.38:.1f}" y2="{cy}" class="flow"/>'
        )
    # skip connections: encoder level k -> decoder block mirrored
    if m["skips"]:
        for k in range(n - 1):
            xe, he = centers[1 + k]
            xd, hd = centers[n + 1 + (n - 2 - k)]
            top = cy - max(he, hd) / 2 - 22 - (n - 1 - k) * 6
            lines.append(
                f'<path d="M {xe:.1f} {cy - he / 2:.1f} C {xe:.1f} {top:.1f}, {xd:.1f} {top:.1f}, {xd:.1f} {cy - hd / 2:.1f}" class="skip"/>'
            )
            if m["gate"]:
                mx = (xe + xd) / 2
                lines.append(
                    f'<circle cx="{mx:.1f}" cy="{top + 8:.1f}" r="11" class="gate"/>'
                    f'<text x="{mx:.1f}" y="{top + 12:.1f}" text-anchor="middle" class="gt">AG</text>'
                )

    legend = ('<text x="12" y="22" class="lg">blue/purple = encoder (purple = pretrained ResNet18) &nbsp; '
              'green = decoder &nbsp; dashed arcs = skip connections</text>'
              if m["skips"] else
              '<text x="12" y="22" class="lg">blue = encoder &nbsp; green = decoder &nbsp; (no skip connections)</text>')

    css = """
    body{margin:0;background:transparent;font-family:system-ui,sans-serif}
    .blk{opacity:.92;animation:pulse 3s ease-in-out infinite}
    .flow{stroke:#64748b;stroke-width:3;stroke-dasharray:6 8;animation:dash 1s linear infinite}
    .skip{fill:none;stroke:#ef4444;stroke-width:2.5;stroke-dasharray:7 7;animation:dash 1.2s linear infinite}
    .gate{fill:#fde68a;stroke:#d97706;stroke-width:2}
    .gt{font-size:10px;font-weight:700;fill:#92400e}
    .lbl{font-size:12px;fill:#64748b}
    .lg{font-size:12px;fill:#64748b}
    @keyframes dash{to{stroke-dashoffset:-30}}
    @keyframes pulse{0%,100%{opacity:.78}50%{opacity:1}}
    """
    svg = f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">{legend}{"".join(lines)}{"".join(parts)}</svg>'
    return f"<style>{css}</style>{svg}"


# --------------------------------------------------------------------------
# Live info from the real code (optional, never raises)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _try_build(name):
    """Try to get the real model object. Returns None if it cannot be built."""
    try:
        import torch
        from src.infer import best_info, load_model

        info = best_info() or {}
        size = info.get("image_size", 128)
        out = load_model(name, size, torch.device("cpu"))
        if isinstance(out, tuple):
            out = out[0]
        if hasattr(out, "parameters"):
            return out
    except Exception:
        pass
    return None


def _param_count(model):
    return sum(p.numel() for p in model.parameters())


def _child_table(model):
    rows = []
    for cname, child in model.named_children():
        total = sum(p.numel() for p in child.parameters())
        rows.append({"Block": cname, "Type": child.__class__.__name__, "Parameters": f"{total:,}"})
    return pd.DataFrame(rows)


def _load_results():
    path = ROOT / "results" / "full_comparison.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------
# Tab 1: Architecture
# --------------------------------------------------------------------------
def render_architecture_tab():
    st.header("Model architectures")
    st.write(
        "We trained and compared **6 models**. Every model takes the grayscale **L** channel "
        "(1 x 128 x 128) and predicts the color channels **a, b** in Lab space. "
        "The original L is kept untouched and recombined with the predicted a, b at the end."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Models compared", "6")
    c2.metric("Best model", "ResNet18 U-Net")
    c3.metric("Best PSNR", BEST_NUMBERS["PSNR"])

    t_overview, t_models, t_relation, t_best = st.tabs(
        ["Pipeline", "Each model", "How they relate", "Best model"]
    )

    with t_overview:
        st.subheader("End-to-end pipeline")
        st.markdown(
            "1. Upload a photo and convert it to Lab. The **L** channel is the grayscale image.\n"
            "2. Resize L to 128 x 128 and feed it to the chosen network.\n"
            "3. The network predicts **a** and **b**.\n"
            "4. Upsample a, b to the original size, attach the original full-resolution L, convert Lab to RGB.\n"
            "5. Output keeps all detail of the input because brightness is never altered."
        )
        components.html(
            """
            <style>
            body{margin:0;font-family:system-ui,sans-serif}
            .row{display:flex;align-items:center;justify-content:space-between;gap:6px}
            .box{flex:1;padding:14px 6px;border-radius:10px;color:#fff;text-align:center;font-size:13px;font-weight:600}
            .arr{font-size:22px;color:#64748b;animation:nudge 1s ease-in-out infinite}
            @keyframes nudge{0%,100%{transform:translateX(0)}50%{transform:translateX(6px)}}
            </style>
            <div class="row">
              <div class="box" style="background:#94a3b8">Photo</div><div class="arr">&#10140;</div>
              <div class="box" style="background:#64748b">L channel</div><div class="arr">&#10140;</div>
              <div class="box" style="background:#3b82f6">U-Net model</div><div class="arr">&#10140;</div>
              <div class="box" style="background:#f59e0b">Predicted a, b</div><div class="arr">&#10140;</div>
              <div class="box" style="background:#10b981">L + a, b &rarr; RGB</div>
            </div>
            """,
            height=90,
        )

    with t_models:
        keys = [m["key"] for m in MODELS]
        titles = {m["title"]: m for m in MODELS}
        choice = st.selectbox("Choose a model", list(titles.keys()))
        m = titles[choice]

        components.html(_diagram_html(m), height=360)

        st.markdown(f"**How it works:** {m['idea']}")
        st.markdown(f"**What it adds:** {m['adds']}")
        st.markdown(f"**Limitation:** {m['weak']}")
        st.markdown(f"**Training loss:** {m['loss']}")

        model = _try_build(m["key"])
        if model is not None:
            st.markdown(f"**Parameters (from the real model):** {_param_count(model):,}")
            df = _child_table(model)
            if not df.empty:
                st.markdown("**Top-level blocks**")
                st.dataframe(df, use_container_width=True, hide_index=True)
            with st.expander("Full layer listing"):
                st.code(str(model), language="text")
        else:
            st.caption("Live layer listing is not available for this model in this session.")

    with t_relation:
        st.subheader("How the models build on each other")
        st.markdown(
            "Each step changes one thing, so the comparison stays fair "
            "(same data, same splits, same optimizer, same 8 epochs, same seed)."
        )
        components.html(
            """
            <style>
            body{margin:0;font-family:system-ui,sans-serif}
            .col{display:flex;flex-direction:column;gap:8px}
            .step{display:flex;align-items:center;gap:10px}
            .nm{min-width:230px;padding:10px;border-radius:10px;color:#fff;font-weight:600;font-size:13px}
            .ch{font-size:13px;color:#64748b;animation:fade 2.4s ease-in-out infinite}
            @keyframes fade{0%,100%{opacity:.55}50%{opacity:1}}
            </style>
            <div class="col">
              <div class="step"><div class="nm" style="background:#64748b">Autoencoder</div><div class="ch">baseline, no shortcuts</div></div>
              <div class="step"><div class="nm" style="background:#3b82f6">Classic U-Net</div><div class="ch">+ skip connections</div></div>
              <div class="step"><div class="nm" style="background:#6366f1">Attention U-Net</div><div class="ch">+ attention gates on skips</div></div>
              <div class="step"><div class="nm" style="background:#8b5cf6">Pix2Pix-style U-Net</div><div class="ch">+ much deeper and wider (41.8M params)</div></div>
              <div class="step"><div class="nm" style="background:#10b981">ResNet18 U-Net &#9733;</div><div class="ch">+ pretrained ImageNet encoder &rarr; best</div></div>
              <div class="step"><div class="nm" style="background:#f59e0b">ResNet18 U-Net (classification)</div><div class="ch">same network, 484-bin classification loss for vivid color</div></div>
            </div>
            """,
            height=340,
        )

        summary = pd.DataFrame(
            [
                {
                    "Model": m["title"].split(". ", 1)[1],
                    "Skip connections": "Yes" if m["skips"] else "No",
                    "Attention": "Yes" if m["gate"] else "No",
                    "Pretrained": "Yes" if m["pretrained"] else "No",
                    "Loss": m["loss"],
                }
                for m in MODELS
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        res = _load_results()
        if res is not None:
            st.markdown("**Measured results (from results/full_comparison.csv)**")
            st.dataframe(res, use_container_width=True, hide_index=True)

    with t_best:
        st.subheader("Best model: ResNet18 U-Net")
        b1, b2, b3 = st.columns(3)
        b1.metric("PSNR", BEST_NUMBERS["PSNR"])
        b2.metric("SSIM", BEST_NUMBERS["SSIM"])
        b3.metric("L1 error", BEST_NUMBERS["L1 error"])
        st.markdown(
            "It won on every pixel metric. Two lessons from the comparison:\n"
            "- A **pretrained encoder** helps a lot when the dataset is small (3,500 training images).\n"
            "- A **bigger model is not automatically better**: the large Pix2Pix-style U-Net finished last.\n\n"
            "Classic U-Net, Attention U-Net and the Autoencoder were within about 0.04 dB of each other, "
            "so on our data they are effectively tied. The classification variant (model 6) scores lower on "
            "PSNR/SSIM (23.05 dB, 0.908) by design, because it commits to vivid colors instead of safe averages."
        )


# --------------------------------------------------------------------------
# Tab 2: Why U-Net?
# --------------------------------------------------------------------------
def render_why_unet_tab():
    st.header("Why did we use U-Net?")
    st.markdown(
        "Colorization is an **image-to-image** problem: the input is an image (L) and the output is an image "
        "of the same size (a, b). U-Net is built for exactly this kind of dense, per-pixel prediction."
    )

    st.subheader("Main reasons")
    st.markdown(
        "1. **Skip connections keep detail.** Edges and textures in the grayscale image are copied directly to the "
        "decoder, so color boundaries line up with object boundaries.\n"
        "2. **Works with small datasets.** We had only 3,500 training images. U-Net is known to train well on limited data.\n"
        "3. **Fully convolutional.** No fixed-size dense layers, so it handles spatial structure naturally.\n"
        "4. **Easy to extend.** We could swap in an attention version, a deeper version, or a pretrained ResNet18 "
        "encoder without changing the overall design. That made a fair controlled comparison possible.\n"
        "5. **Established for this task.** U-Net style generators are used in published image-to-image and "
        "colorization work (for example pix2pix)."
    )

    st.subheader("Alternatives and why we did not pick them")
    st.dataframe(
        pd.DataFrame(
            [
                {"Option": "Plain autoencoder", "Verdict": "Included as our baseline. No skips, so fine detail passes through a bottleneck."},
                {"Option": "GAN", "Verdict": "Not used. Needs more data and careful tuning, and training is less stable. Listed as future scope only."},
                {"Option": "Transformer-based models", "Verdict": "Usually need much more data and compute than our setup."},
                {"Option": "Hand-colored or rule-based methods", "Verdict": "Subjective and slow; cannot generalize across scenes."},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Questions you may be asked")
    qa = [
        ("Why not a GAN?",
         "We deliberately stayed with U-Net. GANs add a discriminator and unstable training, and need more data than the 3,500 images we used. "
         "A GAN is future scope; none of our results use one. Even the Pix2Pix-style model is only its U-Net generator."),
        ("Did U-Net actually beat the autoencoder?",
         "Honest answer: on our pixel metrics the Autoencoder, Classic U-Net and Attention U-Net were within about 0.04 dB, so they are effectively tied. "
         "The clear gain came from the pretrained ResNet18 encoder. U-Net is still the right design because skip connections preserve detail "
         "and let us plug in a pretrained encoder."),
        ("Why did the biggest model do worst?",
         "The Pix2Pix-style U-Net has about 41.8M parameters. With 3,500 images and 8 epochs it is under-trained. Model size must match data size."),
        ("Why Lab color space?",
         "L is the grayscale input itself, so the network predicts only two channels (a, b). We reattach the original L, which keeps full detail."),
        ("Why are the colors sometimes muted?",
         "L1 loss rewards predicting the average of all plausible colors, and the average is grey. "
         "Model 6 addresses this with 484-bin classification and class weights that favor rare, vivid colors."),
        ("What are the limitations?",
         "Only 3,500 training images, 128-pixel inputs, 8 epochs and one random seed. A single ground truth is used although many colorizations are valid. "
         "Skin and clothing remain hard."),
    ]
    for q, a in qa:
        with st.expander(q):
            st.write(a)


def render_architecture_and_why():
    """Convenience: render both as sub-tabs inside one app tab."""
    a, b = st.tabs(["Architecture", "Why U-Net?"])
    with a:
        render_architecture_tab()
    with b:
        render_why_unet_tab()

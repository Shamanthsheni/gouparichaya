"""GouParichaya: explainable desi cow breed recognition with a B-cos ResNet-50 (Streamlit app)."""
import json, math, re
from pathlib import Path
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

# Hugging Face model repository that stores the trained weights
HF_MODEL_REPO = "shamanth2005/gouparichaya-model"
MODEL_FILE, INFO_FILE = "bcos_resnet50_all50.pt", "breed_info_v3.json"

# Breed classes (in training order) and the "Not sure" threshold of the deployed all-50-breeds model
CLASS_NAMES = ["Amritmahal", "Ayrshire", "Bargur", "Dangi", "Deoni", "Gir", "Hallikar", "Hariana", "Himachali Pahari", "Kangayam", "Kankrej", "Kenkatha", "Khariar", "Khillari", "Konkan Kapila", "Kosali", "Krishna_Valley", "Ladakhi", "Lakhimi", "Malnad_gidda", "Mewati", "Nari", "Nimari", "Ongole", "Poda Thirupu", "Pulikulam", "Punganur", "Purnea", "Rathi", "Red kandhari", "Red_Sindhi", "Sahiwal", "Shweta Kapila", "Tharparkar", "Umblachery", "Vechur", "bachaur", "badri", "bhelai", "dagri", "gangatari", "gaolao", "ghumsari", "kherigarh", "malvi", "motu", "nagori", "ponwar", "siri", "thutho"]
THRESHOLD = 0.61

st.set_page_config(page_title="GouParichaya", page_icon="🐄", layout="wide")

def get_file(name):
    """Use a local copy if present, otherwise download it from the Hugging Face model repository."""
    if Path(name).exists():
        return name
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_MODEL_REPO, filename=name)

@st.cache_resource(show_spinner="Loading the model (first visit takes about a minute)...")
def load():
    import bcos
    from bcos.modules import BcosConv2d
    from bcos.data.transforms import AddInverse
    classes, thresh = CLASS_NAMES, THRESHOLD
    model = bcos.pretrained.resnet50_long(pretrained=False)
    model.fc = BcosConv2d(model.fc.linear.in_channels, len(classes), kernel_size=1, b=2)
    model.logit_layer.logit_bias = -math.log(len(classes) - 1)
    model.load_state_dict(torch.load(get_file(MODEL_FILE), map_location="cpu"))
    model.eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), AddInverse()])
    info = json.load(open(INFO_FILE, encoding="utf-8"))
    return model, tf, classes, thresh, info

model, TF, CLASSES, THRESH, BREED_INFO = load()

def bkey(name): return re.sub(r"[^a-z0-9]", "", name.lower())
def pretty(name): return BREED_INFO.get(bkey(name), {}).get("name", name.replace("_", " ").title())

def info_card(name):
    d = BREED_INFO.get(bkey(name))
    if not d: return "_Breed details not available._"
    rows = [("Home tract", d["home_tract"]), ("Main use", d["main_use"]),
            ("How to recognise", d["features"]), ("Notable", d["notes"])]
    body = "\n\n".join(f"**{k}:** {v}" for k, v in rows if v)
    return f"### About {d['name']}\n\n{body}\n\n<sub>Source: {d['source']}</sub>"

def short_line(name):
    d = BREED_INFO.get(bkey(name))
    return f"**{pretty(name)}**: {d['home_tract']}; {d['main_use'].lower()}" if d else f"**{pretty(name)}**"

def probs(img):
    x = TF(img)[None]
    with torch.no_grad():   # average the image and its mirror image
        return ((model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)) / 2)[0].numpy()

def explain(img, idx):
    x = TF(img)[None].requires_grad_()
    e = np.asarray(model.explain(x, idx=int(idx))["explanation"])
    rgba = Image.fromarray((e * 255).astype(np.uint8), "RGBA")
    bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255)); bg.alpha_composite(rgba)
    return bg.convert("RGB")

st.title("🐄 GouParichaya: Know Your Desi Cow")
st.write("Upload a cow photo. An interpretable **B-cos network** identifies the breed among 50 Indian cattle breeds, "
         "shows which parts of the image it used, and tells you about the breed. "
         "It says **Not sure** when it cannot tell reliably.")

tab_up, tab_cam = st.tabs(["Upload a photo", "Use camera"])
with tab_up:
    up = st.file_uploader("Cow photo", type=["jpg", "jpeg", "png", "webp"])
with tab_cam:
    cam = st.camera_input("Take a photo")
src = up or cam

if src is not None:
    img = Image.open(src).convert("RGB")
    with st.spinner("Identifying breed..."):
        p = probs(img); top = p.argsort()[::-1][:3]
        heat = explain(img, top[0])
    sure = p[top[0]] >= THRESH

    c1, c2 = st.columns(2)
    with c1:
        st.image(img, caption="Your photo", width="stretch")
    with c2:
        if sure:
            st.success(f"## {pretty(CLASSES[top[0]])}  ({p[top[0]]:.0%} confident)")
        else:
            st.warning("## Not sure\nThis may be a crossbred, a non-descript desi cow, "
                       "or a breed the model was not trained on.")
        st.markdown("**Top 3 breeds" + ("**" if sure else " (closest matches, low confidence)**"))
        for i in top:
            st.progress(float(p[i]), text=f"{pretty(CLASSES[i])}: {p[i]:.0%}")

    c3, c4 = st.columns(2)
    with c3:
        st.image(heat, caption="B-cos explanation: the image parts the model used", width="stretch")
    with c4:
        if sure:
            st.markdown(info_card(CLASSES[top[0]]), unsafe_allow_html=True)
        else:
            st.markdown("**Closest matches:**\n\n" + "\n\n".join(short_line(CLASSES[i]) for i in top[:2]))

st.caption("Research prototype. Breed facts compiled from NDDB / ICAR-NBAGR sources. "
           "Predictions can be wrong; confirm with a veterinarian or livestock officer.")

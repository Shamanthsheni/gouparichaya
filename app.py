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

st.set_page_config(page_title="GouParichaya – Know your desi cow", page_icon="🐄", layout="wide")

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


# ----------------------------------------------------------------- look and feel
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hind:wght@400;500;600&family=Tiro+Devanagari+Hindi&display=swap');
:root{
  --ink:#1F2A5C; --ink-soft:#4A5378; --haldi:#E0A419; --haldi-wash:#FBF1D9;
  --neem:#3D7A4F; --neem-wash:#E6F0E8; --paper:#F4F6F3; --line:#DCE0E6; --white:#FFFFFF;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{background:var(--paper)!important;}
[data-testid="stAppViewContainer"] *{font-family:'Hind',sans-serif;}
[data-testid="stIconMaterial"], [class*="material-symbols"], [class*="material-icons"]{font-family:'Material Symbols Rounded'!important;}
[data-testid="stHeader"]{background:transparent!important;}
#MainMenu, footer, [data-testid="stToolbar"]{visibility:hidden;}
.block-container{max-width:1120px!important;padding-top:1.2rem!important;padding-bottom:3rem!important;}
p, li, label, span, div{color:var(--ink);}

/* masthead */
.gp-mast{background:var(--ink);border-radius:18px;padding:34px 40px 30px;margin-bottom:28px;
  display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap;}
.gp-mast h1 *, .gp-mast h1 a{color:#fff!important;}
.gp-mast h1 a{display:none!important;}
.gp-mast h1{font-family:'Tiro Devanagari Hindi',serif!important;color:#fff!important;font-size:3.1rem!important;
  line-height:1.05!important;margin:0!important;padding:0!important;font-weight:400!important;letter-spacing:.5px;}
.gp-mast .deva{font-family:'Tiro Devanagari Hindi',serif;color:var(--haldi);font-size:1.5rem;margin-top:6px;}
.gp-mast p{color:#D5DAEC!important;max-width:540px;font-size:1.05rem;line-height:1.55;margin:0;}

/* section headings */
.gp-h{font-family:'Tiro Devanagari Hindi',serif;font-size:1.45rem;color:var(--ink);margin:34px 0 4px;}
.gp-sub{color:var(--ink-soft);font-size:.98rem;margin:0 0 14px;max-width:640px;line-height:1.5;}

/* uploader and tabs */
[data-testid="stFileUploaderDropzone"]{background:var(--white)!important;border:1.5px dashed #AEB6CF!important;border-radius:14px!important;}
[data-testid="stFileUploaderDropzone"] *{color:var(--ink)!important;}
[data-testid="stFileUploaderDropzone"] button{border:1px solid var(--ink)!important;background:var(--white)!important;}
button[data-baseweb="tab"] p{font-size:1rem!important;color:var(--ink-soft)!important;}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--ink)!important;font-weight:600!important;}
[data-baseweb="tab-highlight"]{background:var(--haldi)!important;}

/* verdict */
.gp-verdict{border-radius:16px;padding:26px 28px;border:1px solid var(--line);background:var(--white);}
.gp-verdict.sure{border-left:8px solid var(--neem);}
.gp-verdict.unsure{border-left:8px solid var(--haldi);background:var(--haldi-wash);}
.gp-kicker{font-size:.95rem;color:var(--ink-soft);margin:0 0 2px;}
.gp-breed{font-family:'Tiro Devanagari Hindi',serif;font-size:3rem;line-height:1.1;color:var(--ink);margin:0 0 6px;}
.gp-note{font-size:1rem;line-height:1.5;color:var(--ink);margin:8px 0 0;}
.gp-meter{height:10px;background:#E8EBF0;border-radius:6px;overflow:hidden;margin:10px 0 4px;}
.gp-meter>div{height:100%;border-radius:6px;}
.gp-conf{font-size:.95rem;color:var(--ink-soft);}

/* alternatives */
.gp-alt{display:grid;grid-template-columns:1fr 52px;gap:4px 12px;align-items:center;margin-top:22px;}
.gp-alt .n{font-size:1rem;}
.gp-alt .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink-soft);}
.gp-alt .b{grid-column:1/3;height:6px;background:#E8EBF0;border-radius:4px;overflow:hidden;margin-bottom:8px;}
.gp-alt .b>div{height:100%;background:var(--ink);opacity:.75;}

/* photos */
.gp-fig{margin:0;}
.gp-fig img{width:100%;border-radius:14px;display:block;border:1px solid var(--line);}
.gp-fig figcaption{font-size:.92rem;color:var(--ink-soft);margin-top:8px;line-height:1.45;}
.gp-fig.dark img{background:#000;}

/* registry card */
.gp-card{background:var(--white);border:1px solid var(--line);border-radius:16px;padding:8px 28px 18px;}
.gp-card h3{font-family:'Tiro Devanagari Hindi',serif!important;font-weight:400!important;font-size:1.7rem!important;margin:18px 0 6px!important;color:var(--ink)!important;}
.gp-row{display:grid;grid-template-columns:170px 1fr;gap:16px;padding:12px 0;border-top:1px solid var(--line);}
.gp-row:first-child{border-top:none;}
.gp-row dt{color:var(--ink-soft);font-size:.97rem;}
.gp-row dd{margin:0;font-size:1.02rem;line-height:1.55;}
.gp-src{font-size:.85rem;color:var(--ink-soft);margin-top:10px;}
.gp-matches{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px;}
.gp-match{background:var(--white);border:1px solid #E7D3A0;border-radius:12px;padding:10px 14px;min-width:200px;}
.gp-match b{font-family:'Tiro Devanagari Hindi',serif;font-weight:400;font-size:1.2rem;display:block;}
.gp-match span{font-size:.92rem;color:var(--ink-soft);}

/* empty state */
.gp-tips{background:var(--white);border:1px solid var(--line);border-radius:16px;padding:22px 26px;}
.gp-tips h4{font-family:'Tiro Devanagari Hindi',serif!important;font-weight:400!important;font-size:1.3rem!important;margin:0 0 10px!important;}
.gp-tips ul{margin:0;padding-left:20px;} .gp-tips li{margin:6px 0;line-height:1.5;}
.gp-foot{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);font-size:.88rem;color:var(--ink-soft);line-height:1.5;}
@media (max-width:640px){
  .gp-mast{padding:26px 22px;} .gp-mast h1{font-size:2.3rem!important;}
  .gp-breed{font-size:2.3rem;} .gp-row{grid-template-columns:1fr;gap:2px;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="gp-mast">
  <div><h1>GouParichaya</h1><div class="deva">गौ परिचय</div></div>
  <p>Photograph a cow to find its breed among 50 Indian breeds. The app shows what it looked at,
  and tells you plainly when it is not sure.</p>
</div>
""", unsafe_allow_html=True)


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


import base64, html, io
E = html.escape

def data_uri(img, max_side=900):
    im = img.copy(); im.thumbnail((max_side, max_side))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def figure(img, caption, dark=False):
    return (f'<figure class="gp-fig{" dark" if dark else ""}"><img src="{data_uri(img)}" alt="{E(caption)}">'
            f'<figcaption>{E(caption)}</figcaption></figure>')

def registry_card(name):
    d = BREED_INFO.get(bkey(name))
    if not d:
        return ""
    rows = [("Home tract", d["home_tract"]), ("Main use", d["main_use"]),
            ("How to recognise", d["features"]), ("Worth knowing", d["notes"])]
    body = "".join(f'<div class="gp-row"><dt>{k}</dt><dd>{E(v)}</dd></div>' for k, v in rows if v)
    return (f'<div class="gp-card"><dl style="margin:6px 0 0">{body}</dl>'
            f'<div class="gp-src">Source: {E(d["source"])}</div></div>')

def match_chip(name, p):
    d = BREED_INFO.get(bkey(name), {})
    tract = E(d.get("home_tract", "")); use = E(d.get("main_use", ""))
    return (f'<div class="gp-match"><b>{E(pretty(name))}</b><span>{tract}<br>{use}</span>'
            f'<span style="float:right;font-variant-numeric:tabular-nums">{p:.0%}</span></div>')

model, TF, CLASSES, THRESH, BREED_INFO = load()

left, right = st.columns([1, 1], gap="large")
with left:
    tab_up, tab_cam = st.tabs(["Upload a photo", "Take a photo"])
    with tab_up:
        up = st.file_uploader("Cow photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    with tab_cam:
        cam = st.camera_input("Camera", label_visibility="collapsed")
src = up or cam

if src is None:
    with right:
        st.markdown("""
<div class="gp-tips"><h4>For the best result</h4><ul>
<li>Show the whole animal, ideally from the side.</li>
<li>Take the photo in daylight, without people or other animals in front.</li>
<li>One animal per photo. Crossbred cows will usually get <b>Not sure</b>, which is the honest answer.</li>
</ul></div>""", unsafe_allow_html=True)
else:
    img = Image.open(src).convert("RGB")
    with st.spinner("Looking at the photo..."):
        p = probs(img); top = p.argsort()[::-1][:3]
        heat = explain(img, top[0])
    best, conf = CLASSES[top[0]], float(p[top[0]])
    sure = conf >= THRESH
    colour = "var(--neem)" if sure else "var(--haldi)"

    with left:
        st.markdown(figure(img, "Your photo"), unsafe_allow_html=True)
    with right:
        if sure:
            head = (f'<p class="gp-kicker">This looks like</p><div class="gp-breed">{E(pretty(best))}</div>'
                    f'<div class="gp-meter"><div style="width:{conf*100:.0f}%;background:{colour}"></div></div>'
                    f'<div class="gp-conf">{conf:.0%} confidence</div>')
        else:
            head = (f'<div class="gp-breed">Not sure</div>'
                    f'<p class="gp-note">The model cannot name this breed reliably. The animal may be crossbred, '
                    f'a non-descript desi cow, or a breed it was not trained on.</p>'
                    f'<div class="gp-meter"><div style="width:{conf*100:.0f}%;background:{colour}"></div></div>'
                    f'<div class="gp-conf">Highest confidence {conf:.0%}, below the {THRESH:.0%} needed to answer</div>')
        shown = top[1:] if sure else top
        alts = "".join(f'<div class="n">{E(pretty(CLASSES[i]))}</div><div class="v">{p[i]:.0%}</div>'
                       f'<div class="b"><div style="width:{p[i]*100:.0f}%"></div></div>' for i in shown)
        label = "Other possibilities" if sure else "Closest breeds"
        st.markdown(f'<div class="gp-verdict {"sure" if sure else "unsure"}">{head}'
                    f'<p class="gp-kicker" style="margin-top:22px">{label}</p><div class="gp-alt" style="margin-top:6px">{alts}</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="gp-h">Why this answer</div><p class="gp-sub">GouParichaya uses a B-cos network, '
                'which can show exactly which parts of the photo added up to its answer. Bright areas mattered most; '
                'dark areas were ignored.</p>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(figure(img.resize((448, 448)), "Photo as the model sees it"), unsafe_allow_html=True)
    with b:
        st.markdown(figure(heat.resize((448, 448)), f"Evidence for {pretty(best)}", dark=True), unsafe_allow_html=True)

    if sure:
        st.markdown(f'<div class="gp-h">About the {E(pretty(best))} breed</div>', unsafe_allow_html=True)
        st.markdown(registry_card(best), unsafe_allow_html=True)
    else:
        st.markdown('<div class="gp-h">Breeds it might be close to</div><p class="gp-sub">Compare these with the '
                    'animal, or ask a veterinarian or livestock officer to confirm.</p>', unsafe_allow_html=True)
        st.markdown('<div class="gp-matches">' + "".join(match_chip(CLASSES[i], p[i]) for i in top[:2]) + "</div>",
                    unsafe_allow_html=True)

st.markdown('<div class="gp-foot">GouParichaya is a student research project (BAI701, Jyothy Institute of Technology). '
            'Predictions can be wrong; confirm important decisions with a veterinarian. Breed facts are compiled from '
            'NDDB and ICAR-NBAGR sources.</div>', unsafe_allow_html=True)

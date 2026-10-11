import json, os, sqlite3
from datetime import datetime
import cv2, numpy as np, torch, streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = f"{BASE}/models/best.pt"
DB = f"{BASE}/skin_history.db"
RECS = json.load(open(f"{BASE}/recs.json"))
DISCLAIMER = "This is general skincare advice, not a medical diagnosis."

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def severity(result, names):
    h, w = result.orig_shape
    stats = {}
    for b in result.boxes:
        c = names[int(b.cls)]
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        d = stats.setdefault(c, {"count": 0, "area": 0.0})
        d["count"] += 1
        d["area"] += (x2 - x1) * (y2 - y1)
    out = {}
    for c, d in stats.items():
        pct = 100 * d["area"] / (h * w)
        if c in ("Acne", "spots", "pigmentation"):
            lvl = "mild" if d["count"] <= 5 else "moderate" if d["count"] <= 15 else "severe"
        else:
            lvl = "mild" if pct < 2 else "moderate" if pct < 5 else "severe"
        out[c] = {"count": d["count"], "area_pct": round(pct, 2), "severity": lvl}
    return out

def heatmap(model, bgr, size=640):
    img = cv2.resize(bgr, (size, size))
    rgb = img[:, :, ::-1].astype(np.float32) / 255
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)
    net = model.model.eval()
    dev = next(net.parameters()).device
    acts = {}
    hook = net.model[-2].register_forward_hook(lambda m, i, o: acts.update(a=o.detach()))
    with torch.no_grad():
        net(x.to(dev).float())
    hook.remove()
    a = acts["a"][0].float().cpu().numpy()
    C, H, W = a.shape
    flat = a.reshape(C, -1).T
    flat = flat - flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    cam = (flat @ vt[0]).reshape(H, W)
    if abs(cam.min()) > cam.max():
        cam = -cam
    cam = cv2.resize(np.maximum(cam, 0), (size, size))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.cvtColor(cv2.addWeighted(img, 0.55, heat, 0.45, 0), cv2.COLOR_BGR2RGB)

def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT, scan_date TEXT, condition TEXT, count INTEGER, area_pct REAL, severity TEXT)""")
    return con

def save_scan(user, sev):
    con = db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for c, v in sev.items():
        con.execute("INSERT INTO scans (user,scan_date,condition,count,area_pct,severity) VALUES (?,?,?,?,?,?)",
                    (user, now, c, v["count"], v["area_pct"], v["severity"]))
    con.commit()

def compare(user):
    con = db()
    dates = [d[0] for d in con.execute(
        "SELECT DISTINCT scan_date FROM scans WHERE user=? ORDER BY scan_date DESC LIMIT 2", (user,))]
    if len(dates) < 2:
        return None
    load = lambda d: {c: a for c, a in con.execute(
        "SELECT condition, area_pct FROM scans WHERE user=? AND scan_date=?", (user, d))}
    new, old = load(dates[0]), load(dates[1])
    rows = []
    for c in sorted(set(new) | set(old)):
        diff = round(new.get(c, 0) - old.get(c, 0), 2)
        rows.append({"condition": c, "previous area %": old.get(c, 0), "latest area %": new.get(c, 0),
                     "change": "improved" if diff < -0.2 else "worsened" if diff > 0.2 else "stable"})
    return dates, rows

st.set_page_config(page_title="Skin Analysis", layout="wide")
st.title("Multi-condition facial skin analysis")
st.caption(DISCLAIMER)

model = load_model()
user = st.sidebar.text_input("Your name / ID", "demo")
conf = st.sidebar.slider("Detection confidence", 0.05, 0.9, 0.25)
file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])

if file:
    rgb = np.array(Image.open(file).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    res = model.predict(bgr, conf=conf, verbose=False)[0]
    sev = severity(res, model.names)

    c1, c2 = st.columns(2)
    c1.subheader("Detections")
    c1.image(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
    c2.subheader("Model attention (EigenCAM)")
    c2.image(heatmap(model, bgr), use_container_width=True)

    st.subheader("Severity and recommendations")
    if not sev:
        st.success("No visible conditions detected.")
    for c, v in sev.items():
        with st.expander(f"{c}: {v['count']} found, {v['severity']} ({v['area_pct']}% of image)", expanded=True):
            for t in RECS.get(c, {}).get(v["severity"], []):
                st.write("•", t)
    if sev and st.button("Save this scan to my history"):
        save_scan(user, sev)
        st.success("Saved.")

st.subheader("Progress over time")
cmp = compare(user)
if cmp:
    st.write(f"Comparing {cmp[0][1]} to {cmp[0][0]}")
    st.table(cmp[1])
else:
    st.info("Save at least 2 scans to see changes.")

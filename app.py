import streamlit as st
import cv2
import time
import os
import torch
import numpy as np
import tempfile
from engine import DetectionEngine, get_available_models, download_model_from_url, save_uploaded_model, list_compute_devices
from streamer import MultiCameraStreamer
from state_manager import StateManager

# UI Config
st.set_page_config(page_title="Vision Web Node Pro", layout="wide", initial_sidebar_state="expanded")

state = StateManager()
streamer = MultiCameraStreamer()

# 🔐 AUTH
if 'auth' not in st.session_state: st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔐 Vision Login")
    with st.form("login"):
        u = st.text_input("User")
        p = st.text_input("Pass", type="password")
        if st.form_submit_button("Login"):
            if u in state.data["web_users_auth"] and state.data["web_users_auth"][u] == p:
                st.session_state.auth = True
                st.session_state.user = u
                state.manage_user(u, "login")
                st.rerun()
            else: st.error("Access Denied")
    st.stop()

user = st.session_state.user
role = state.data["user_sessions"].get(user, {}).get("role", "view")

# 📹 WEB SIDEBAR (Full Control)
with st.sidebar:
    st.title(f"👤 {user} ({role})")

    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"
    theme_choice = st.selectbox("Theme", ["Dark", "Light"], index=0 if st.session_state.theme == "Dark" else 1)
    st.session_state.theme = theme_choice
    
    # Sources Management
    st.subheader("📷 Джерела")
    new_url = st.text_input("Додати URL (IP/RTSP):")
    if st.button("Додати URL"):
        if new_url: state.add_saved_source(new_url); st.success("OK")

    folder_path = st.text_input("Додати папку (локальна/мережева):")
    if st.button("Додати папку"):
        if folder_path and os.path.isdir(folder_path):
            state.add_saved_source(folder_path); st.success("Папку додано")
        elif folder_path:
            st.error("Папку не знайдено")
    
    up_file = st.file_uploader(
        "Завантажити файл (Video/Img):",
        type=['mp4','avi','mkv','mov','wmv','webm','m4v','mpeg','mpg','jpg','jpeg','png','bmp','tiff','tif','webp']
    )
    if up_file:
        t = tempfile.NamedTemporaryFile(delete=False, suffix='.'+up_file.name.split('.')[-1])
        t.write(up_file.read())
        state.add_saved_source(t.name)
        st.success("Файл додано")

    st.divider()
    # Models Management
    st.subheader("🧠 Моделі")
    model_url = st.text_input("URL моделі (.pt/.pth)")
    if st.button("Завантажити модель"):
        if model_url:
            try:
                target = download_model_from_url(model_url)
                st.success(f"Модель збережено: {target}")
            except Exception as exc:
                st.error(f"Помилка завантаження: {exc}")

    model_file = st.file_uploader("Завантажити модель (.pt/.pth)", type=['pt', 'pth'])
    if model_file:
        try:
            target = save_uploaded_model(model_file.getvalue(), model_file.name)
            st.success(f"Модель збережено: {target}")
        except Exception as exc:
            st.error(f"Помилка збереження: {exc}")

    st.divider()
    # Slot Config (for current user view)
    st.subheader("⚙️ Налаштування вікна")
    slots = state.data["slot_configs"]
    sel_slot = st.selectbox("Оберіть камеру:", [f"Slot {s['id']+1}" for s in slots])
    slot_id = int(sel_slot.split(" ")[1]) - 1
    cfg = next(s for s in slots if s["id"] == slot_id)

    sources = state.data["saved_sources"]
    models = get_available_models()
    src_index = sources.index(cfg["src"]) if cfg["src"] in sources else 0
    model_index = models.index(cfg["model"]) if cfg["model"] in models else 0

    src_choice = st.selectbox("Джерело", sources, index=src_index)
    model_choice = st.selectbox("Модель", models, index=model_index)

    v_mode = st.selectbox("Mode:", ["Live", "Heatmap"], index=0)
    v_qual = st.selectbox("Quality:", ["480p", "720p", "360p"], index=0)

    devices = list_compute_devices()
    device_ids = [d[0] for d in devices]
    device_labels = [d[1] for d in devices]
    device_key = f"slot_device_{slot_id}"
    if device_key not in st.session_state:
        st.session_state[device_key] = "auto"
    device_index = device_ids.index(st.session_state[device_key]) if st.session_state[device_key] in device_ids else 0
    device_choice_label = st.selectbox("Device", device_labels, index=device_index)
    device_choice = device_ids[device_labels.index(device_choice_label)]

    col_a, col_b, col_c = st.columns(3)
    if col_a.button("▶️ Start"):
        if os.path.isdir(src_choice):
            st.error("Папку не можна запускати як live-стрім. Використайте desktop batch-детекцію.")
        else:
            st.session_state[device_key] = device_choice
            state.update_slot(slot_id, {"src": src_choice, "model": model_choice, "running": True, "mode": v_mode, "quality": v_qual})
            st.success("Запущено")
            st.rerun()
    if col_b.button("⏹ Stop"):
        state.update_slot(slot_id, {"running": False})
        st.warning("Зупинено")
        st.rerun()
    if col_c.button("🔄 Refresh URL"):
        streamer.refresh_stream_url(src_choice, restart=True)
        st.success("URL оновлено")

    st.divider()
    st.subheader("🖥️ Перегляд")
    view_slots = st.multiselect(
        "Оберіть слоти для перегляду",
        [f"Slot {s['id']+1}" for s in slots],
        default=[f"Slot {s['id']+1}" for s in slots if s.get("running")]
    )

# Apply theme styles
theme_css = """
<style>
.stDeployButton {display:none;}
.stButton>button {width: 100%;}
img {border-radius: 10px; border: 2px solid #333;}
</style>
"""
if st.session_state.theme == "Light":
    theme_css = """
    <style>
    .stDeployButton {display:none;}
    .stButton>button {width: 100%;}
    .stApp {background-color: #f4f4f4; color: #111;}
    img {border-radius: 10px; border: 2px solid #bbb;}
    </style>
    """
st.markdown(theme_css, unsafe_allow_html=True)

# 📺 MAIN VIEW
st.title("📡 Vision Real-Time Feed")

view_slot_ids = [int(v.split(" ")[1]) - 1 for v in view_slots]
if not view_slot_ids:
    st.info("Оберіть слоти для перегляду у сайдбарі.")
    st.stop()

cols_count = 2 if len(view_slot_ids) > 1 else 1
placeholders = {}
caption_slots = {}

for idx, sid in enumerate(view_slot_ids):
    if idx % cols_count == 0:
        cols = st.columns(cols_count)
    col = cols[idx % cols_count]
    placeholders[sid] = col.empty()
    caption_slots[sid] = col.empty()

while True:
    for sid in view_slot_ids:
        cfg = next((s for s in state.data["slot_configs"] if s["id"] == sid), None)
        if not cfg or not cfg.get("running"):
            placeholders[sid].info("Потік зупинений")
            caption_slots[sid].caption("Device: --")
            continue

        device_choice = st.session_state.get(f"slot_device_{sid}", "auto")
        eng_key = f"web_eng_{sid}_{cfg['model']}_{device_choice}"
        if eng_key not in st.session_state:
            st.session_state[eng_key] = DetectionEngine(
                model_path=None if "Mock" in cfg["model"] else cfg["model"],
                device=device_choice
            )
        engine = st.session_state[eng_key]

        stream_key = f"web_{sid}"
        if stream_key not in streamer.running:
            streamer.start(cfg["src"], stream_key=stream_key)

        frame = streamer.get_frame(stream_key)
        if frame is not None:
            processed, _ = engine.process_frame(
                frame, view_type=cfg.get("mode", "Live"), quality=cfg.get("quality", "480p"),
                visibility_map=cfg.get("visibility_map"), night_mode=cfg.get("night_mode"), zoom=cfg.get("zoom", 1.0)
            )
            placeholders[sid].image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            caption_slots[sid].caption(f"Device: {engine.device_label}")
        else:
            placeholders[sid].warning("Connecting...")
            caption_slots[sid].caption(f"Device: {engine.device_label}")
    time.sleep(0.05)

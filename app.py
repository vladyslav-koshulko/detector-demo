import streamlit as st
import cv2
import time
import os
import torch
import numpy as np
from engine import DetectionEngine, get_available_models
from streamer import MultiCameraStreamer
from state_manager import StateManager

# Mobile-First Config
st.set_page_config(page_title="Vision Mobile v4", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stDeployButton {display:none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 600px;}
    img {border-radius: 20px; border: 4px solid #1e1e1e; margin-bottom: 15px; width: 100% !important;}
    .stButton>button {width: 100%; height: 4rem; border-radius: 15px; font-weight: bold; font-size: 1.2rem; background-color: #007BFF; color: white;}
    </style>
    """, unsafe_allow_html=True)

state = StateManager()
streamer = MultiCameraStreamer()

# 🔐 LOGIN / AUTH LOGIC
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

def login_screen():
    st.title("🔐 Vision Login")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Увійти"):
            if u in state.data["web_users_auth"] and state.data["web_users_auth"][u] == p:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Невірний логін або пароль")
    st.stop()

if not st.session_state.logged_in:
    login_screen()

# 🛡️ Access Control from Admin
username = st.session_state.username
status = state.data["user_access"].get(username, "pending")

if status == "blocked":
    st.error("🚫 Доступ заблоковано Адміністратором.")
    st.stop()
elif status == "pending":
    st.warning("⏳ Очікування підтвердження доступу від Адміністратора...")
    st.info(f"Ваш акаунт: {username}. Будь ласка, попросіть адміна надати доступ.")
    time.sleep(5)
    st.rerun()

# 📹 MAIN STREAMING UI
st.title("📡 Vision Mobile")

active_slots = [s for s in state.data["slot_configs"] if s["running"]]

if not active_slots:
    st.info("👋 Камери поки що вимкнені Адміністратором.")
    if st.button("🔄 Оновити"): st.rerun()
else:
    # Navigation
    tab_labels = [f"Камера {s['id']+1}" for s in active_slots]
    selected = st.selectbox("Оберіть камеру:", tab_labels)
    
    s_idx = int(selected.split(" ")[1]) - 1
    # Find exact config by ID
    cfg = next((s for s in active_slots if s["id"] == s_idx), active_slots[0])
    
    # Session Engine Init
    eng_key = f"web_v4_eng_{cfg['id']}_{cfg['model']}"
    if eng_key not in st.session_state:
        m_path = None if "Mock" in cfg["model"] else cfg["model"]
        st.session_state[eng_key] = DetectionEngine(model_path=m_path, threshold=cfg["conf"])
    
    engine = st.session_state[eng_key]
    placeholder = st.empty()
    
    # Loop
    while True:
        # Check if still running in Admin
        fresh_state = StateManager() # reload from file
        cfg_fresh = next((s for s in fresh_state.data["slot_configs"] if s["id"] == cfg["id"]), None)
        if not cfg_fresh or not cfg_fresh["running"]:
            st.warning("⚠️ Потік зупинено.")
            time.sleep(2); st.rerun()

        frame = streamer.get_frame(cfg["src"])
        if frame is not None:
            # Web node respects visibility_map from Admin
            processed, _ = engine.process_frame(
                frame, view_type=cfg["mode"], 
                visibility_map=cfg["visibility_map"],
                night_mode=cfg["night_mode"],
                quality=cfg["quality"]
            )
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            placeholder.image(rgb)
        else:
            placeholder.info("Підключення...")
        
        time.sleep(0.05)

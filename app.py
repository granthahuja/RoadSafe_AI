import os
import cv2
import tempfile
from database.db import (initialize_database, save_incident, get_all_incidents, save_rescue_case, update_rescue_case, get_all_rescue_cases)
from modules.severity import (load_visual_model, analyze_frame, preliminary_assessment,assess_incident)
from modules.rescue_chain import (create_rescue_case, initiate_response, acknowledge_response, confirm_assistance)
import streamlit.components.v1 as components
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
from datetime import datetime
import html
import pandas as pd

#Initialize permanent incident database 
initialize_database()

# Optional browser geolocation component. Install with:
# python -m pip install streamlit-geolocation
try:
    from streamlit_geolocation import streamlit_geolocation
    GEOLOCATION_AVAILABLE = True
except ImportError:
    streamlit_geolocation = None
    GEOLOCATION_AVAILABLE = False

st.set_page_config(page_title="RoadSafe AI | Command Center", page_icon="🚨", layout="wide", initial_sidebar_state="expanded")
MODEL_PATH = Path(__file__).parent / "road_accident_mobilenetv2.keras"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #061225;
    --panel: #0b1b31;
    --line: #1d3554;
    --muted: #91a9c8;
    --cyan: #20d8f5;
    --red: #ff3154;
    --white: #f4f8ff;
}

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: radial-gradient(
        ellipse at 68% 0%,
        #102745 0%,
        #07152a 40%,
        #040d1b 100%
    );
    color: var(--white);
}

[data-testid="stHeader"] {
    background: rgba(4,13,27,.88);
}

/* MAIN SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#07162a,#061020);
    border-right: 1px solid #19304c;
}

[data-testid="stSidebar"] * {
    color: #dce9ff;
}

[data-testid="stSidebar"] section {
    padding-top: 1rem;
}

/* BRAND */
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 20px;
    border-bottom: 1px solid #1a2d46;
    margin-bottom: 18px;
}

.brand-shield {
    width: 44px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid var(--red);
    border-radius: 13px 13px 18px 18px;
    color: var(--cyan);
    font-size: 25px;
}

.brand-name {
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -.7px;
}

.brand-name span {
    color: var(--red);
}

.brand-sub {
    color: #9bb1ce;
    font-size: 10px;
    letter-spacing: .5px;
    margin-top: 4px;
}

/* COMMAND-CENTRE SIDEBAR NAVIGATION */

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 8px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex;
    align-items: center;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 9px;
    padding: 11px 12px;
    margin-bottom: 4px;
    transition: all 0.2s ease;
    cursor: pointer;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #102943 !important;
    border-color: #244563 !important;
    transform: translateX(3px);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(
        90deg,
        rgba(32,216,245,.16),
        rgba(32,216,245,.04)
    ) !important;
    border: 1px solid rgba(32,216,245,.4) !important;
    border-left: 3px solid #20d8f5 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #c5d5e9 !important;
    font-size: 13px;
    font-weight: 600;
    margin: 0;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: #20d8f5 !important;
    font-weight: 800;
}

/* EXPANDABLE SIDEBAR SECTIONS */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #0d2037 !important;
    border: 1px solid #1d3855 !important;
    border-radius: 10px !important;
    margin-bottom: 12px;
    overflow: hidden;
}

[data-testid="stSidebar"] [data-testid="stExpander"] details {
    background: #0d2037 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: #0d2037 !important;
    color: #eaf5ff !important;
    font-weight: 600;
    font-size: 13px;
    padding: 12px;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background: #142d49 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary span,
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    color: #20d8f5 !important;
    fill: #20d8f5 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #0d2037 !important;
    color: #dce9ff !important;
}


/* SIDEBAR CAPTIONS AND DIVIDERS */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #7896b8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}

[data-testid="stSidebar"] hr {
    border-color: #1d3554;
    margin: 16px 0;
}

.footer {
    text-align: center;
    color: #6683a6;
    font-size: 10px;
    padding: 12px 0;
}

/* MAIN CONTENT */
.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

h1, h2, h3, p, label {
    color: var(--white);
}

.topline {
    color: #75f5c2;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .7px;
    text-transform: uppercase;
    margin: 4px 0 20px;
}

.hero-title {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.2;
}

.hero-title em {
    font-style: normal;
    color: var(--cyan);
}

.hero-kicker {
    color: #a8c2e5;
    font-size: 14px;
    margin-top: 7px;
    margin-bottom: 22px;
}

.section-card {
    background: linear-gradient(
        145deg,
        rgba(12,29,51,.96),
        rgba(7,21,39,.96)
    );
    border: 1px solid #1e3858;
    border-radius: 12px;
    padding: 18px 19px;
    margin-bottom: 16px;
    box-shadow: 0 12px 30px rgba(0,0,0,.12);
}

.section-heading {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 3px;
}

.subheading {
    font-size: 11px;
    color: #9cb3d1;
    line-height: 1.5;
}

.eyebrow {
    color: #91aac9;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    font-size: 10px;
    font-weight: 700;
}

.stat-value {
    font-size: 24px;
    font-weight: 800;
    margin: 8px 0 4px;
}

.stat-note {
    color: #94accb;
    font-size: 11px;
    line-height: 1.45;
}

.cyan {
    color: var(--cyan) !important;
}

.status-pill {
    display: inline-flex;
    border-radius: 7px;
    padding: 10px 12px;
    background: rgba(255,49,84,.10);
    border: 1px solid rgba(255,49,84,.28);
    color: #ff5570;
    font-weight: 800;
    font-size: 12px;
}

.status-pill.normal {
    background: rgba(36,227,158,.09);
    border-color: rgba(36,227,158,.24);
    color: #48e7ad;
}

.notice {
    padding: 10px 12px;
    background: #10253e;
    border: 1px solid #294563;
    border-radius: 8px;
    color: #a9c0de;
    font-size: 11px;
    line-height: 1.5;
}

/* BUTTONS */
.stButton > button {
    border-radius: 7px;
    border: 1px solid #244c75;
    background: #14365a;
    color: #eaf5ff;
    font-weight: 700;
    min-height: 42px;
}

.stButton > button:hover {
    border-color: var(--cyan);
    color: #fff;
    background: #17436a;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#ff3154,#f21d49);
    border: 0;
    color: white;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: #091a30;
    border: 1px dashed #20d8f5;
    border-radius: 8px;
    padding: 10px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stFileUploader"] button {
    background: #00D9FF !important;
    color: #06111F !important;
    border: 1px solid #00D9FF !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {
    color: #06111F !important;
}

[data-testid="stFileUploader"] small {
    color: #c4d5e9 !important;
}

/* MAIN CONTENT EXPANDER - DARK THEME FIX */
[data-testid="stMain"] [data-testid="stExpander"] {
    background: #0b1b31 !important;
    border: 1px solid #1d3554 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

[data-testid="stMain"] [data-testid="stExpander"] details {
    background: #0b1b31 !important;
}

[data-testid="stMain"] [data-testid="stExpander"] summary {
    background: #0b1b31 !important;
    color: #20d8f5 !important;
}

[data-testid="stMain"] [data-testid="stExpander"] summary p {
    color: #20d8f5 !important;
}

[data-testid="stMain"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #0b1b31 !important;
    color: #f4f8ff !important;
}

[data-testid="stMain"] [data-testid="stExpander"] [data-testid="stMarkdown"] p {
    color: #f4f8ff !important;
}

/* RESPONSE NETWORK LINK BUTTONS */
[data-testid="stMain"] [data-testid="stLinkButton"] a {
    background: linear-gradient(90deg, #123b50, #164b68) !important;
    color: #38bdf8 !important;
    border: 1px solid #20d8f5 !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    text-decoration: none !important;
}

[data-testid="stMain"] [data-testid="stLinkButton"] a p,
[data-testid="stMain"] [data-testid="stLinkButton"] a span {
    color: #38bdf8 !important;
}

[data-testid="stMain"] [data-testid="stLinkButton"] a:hover {
    background: #174b68 !important;
    border-color: #ffffff !important;
}

/* =====================================
   ROADSAFE AI - X INSPIRED SIDEBAR
===================================== */

[data-testid="stSidebar"] {
    background: #050505 !important;
    border-right: 1px solid #202327 !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: #050505 !important;
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #e7e9ea !important;
}

/* Brand */
[data-testid="stSidebar"] .brand {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 16px 8px 20px;
}

[data-testid="stSidebar"] .brand-shield {
    width: 46px;
    height: 46px;
    border-radius: 15px;
    background: #102b3d;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
}

[data-testid="stSidebar"] .brand-name {
    font-size: 23px;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.8px;
}

[data-testid="stSidebar"] .brand-name span {
    color: #38bdf8;
}

[data-testid="stSidebar"] .brand-sub {
    font-size: 10px;
    color: #8294a8;
    margin-top: 3px;
}

/* Workspace label */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #71767b !important;
    font-size: 10px !important;
    font-weight: 800 !important;
    letter-spacing: 2px;
    padding-left: 10px;
}

/* Navigation container */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 7px !important;
}

/* Navigation options */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent !important;
    border-radius: 30px !important;
    padding: 12px 15px !important;
    transition: background 0.2s ease;
    cursor: pointer;
}

/* Hover effect */
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #181818 !important;
}

/* Hide radio circles */
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

/* Navigation text */
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #d7dbdf !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* Selected navigation item */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #102b3d !important;
    border: 1px solid #164b68 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: #38bdf8 !important;
    font-weight: 800 !important;
}

/* Dividers */
[data-testid="stSidebar"] hr {
    border-color: #202327 !important;
}

/* Expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #101214 !important;
    border: 1px solid #272a2d !important;
    border-radius: 12px !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #e7e9ea !important;
}

/* =====================================
   ROADSAFE AI - FULL DARK THEME
===================================== */

/* Main application background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #090b0f !important;
}

/* Main content area */
[data-testid="stMain"] .block-container {
    background: transparent !important;
    color: #e7edf5 !important;
}

/* Main headings */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #f1f5f9 !important;
}

/* Main paragraph and captions */
[data-testid="stMain"] p,
[data-testid="stMain"] label {
    color: #c1cbd7;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #111820 !important;
    border: 1px solid #263746 !important;
    padding: 15px !important;
    border-radius: 13px !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: #111820 !important;
    border: 1px solid #263746 !important;
    border-radius: 12px !important;
}

/* Input fields */
[data-testid="stMain"] input,
[data-testid="stMain"] textarea,
[data-testid="stMain"] [data-baseweb="select"] > div {
    background: #111820 !important;
    color: #f1f5f9 !important;
    border-color: #263746 !important;
}

/* Main dividers */
[data-testid="stMain"] hr {
    border-color: #263746 !important;
}

/* Main alerts */
[data-testid="stAlert"] {
    background: #111820 !important;
    border-radius: 12px !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background: #0a1a2e;
    border: 1px solid #1c3654;
    padding: 12px;
    border-radius: 9px;
}

[data-testid="stMetricLabel"] p {
    color: #9eb5d2 !important;
    font-size: 12px;
}

[data-testid="stMetricValue"] {
    color: #f4f8ff !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the MobileNetV2 model...")
def load_model():
    return tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
        safe_mode=False
    )


if "roadsafe_history" not in st.session_state:
    st.session_state.roadsafe_history = []

if "roadsafe_last_result" not in st.session_state:
    st.session_state.roadsafe_last_result = None

# Navigation and sidebar
with st.sidebar:

    # Brand
    st.markdown(
        '''
        <div class="brand">
            <div class="brand-shield">🛣</div>
            <div>
                <div class="brand-name">RoadSafe <span>AI</span></div>
                <div class="brand-sub">Safer Roads. Smarter Response.</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Navigation
    st.caption("WORKSPACE")

    page_labels = [
        "⌂  Dashboard",
        "▣  Live Detection",
        "◷  Incident History",
        "⛓  RescueChain",
        "✚  Hospital Dashboard",
        "⌖  Response Network",
        "▥  Model Insights"
    ]

    selected = st.radio(
        "Navigation",
        page_labels,
        key="roadsafe_navigation",
        label_visibility="collapsed"
    )

    page = selected.split("  ", 1)[-1].strip()

    st.markdown("---")

    # System information
    st.caption("SYSTEM INFORMATION")

    with st.expander("Emergency response workflow"):
        stages = [
            "① Image received",
            "② AI analysis",
            "③ Accident assessment",
            "④ Location verification",
            "⑤ Response coordination"
        ]

        for stage in stages:
            st.markdown(stage)

        st.info(
            "GPS, hospital notifications and dispatch "
            "are demo-only until connected to real services."
        )

# Model diagnostics
    with st.expander("Model configuration"):

        single_output_meaning = st.radio(
            "Single sigmoid output represents:",
            ["Normal probability", "Accident probability"],
            index=0,
            help="Based on your original training labels."
        )

        two_output_mapping = st.radio(
            "For two outputs, index 0 means:",
            ["Accident", "Normal"],
            index=0
        )

    st.markdown("---")

    st.markdown(
        '''
        <div class="footer">
            🛡️ Better detection.<br>
            Faster response.<br>
            Safer tomorrow.
        </div>
        ''',
        unsafe_allow_html=True
    )

st.markdown('<div class="topline">🟢 SYSTEM ONLINE &nbsp; • &nbsp; LOCAL DEMO</div>', unsafe_allow_html=True)
history = st.session_state.roadsafe_history
accident_count = sum(1 for x in history if x["accident"])
normal_count = len(history) - accident_count

if page == "Dashboard":

    # =========================
    # COMMAND CENTRE HEADER
    # =========================

    st.markdown(
                """
        <div style="background:linear-gradient(115deg,#102943,#0b1b31 65%,#123a54);border:1px solid #244563;border-radius:18px;padding:28px 30px;margin-bottom:22px;box-shadow:0 12px 35px rgba(0,0,0,.18);">
        <div style="color:#20d8f5;font-size:11px;font-weight:800;letter-spacing:2px;margin-bottom:12px;">ROADSAFE AI / COMMAND CENTRE</div>
        <div style="color:#f4f8ff;font-size:30px;font-weight:800;line-height:1.2;">Intelligent Road Safety Monitoring</div>
        <div style="color:#a9c0de;font-size:14px;margin-top:10px;">Monitor detections, review potential incidents and coordinate emergency response workflows.</div>
        <div style="margin-top:20px;">
        <span style="background:rgba(36,227,158,.12);color:#48e7ad;border:1px solid rgba(36,227,158,.3);padding:7px 12px;border-radius:20px;font-size:11px;font-weight:700;">● SYSTEM ONLINE</span>
        <span style="color:#91a9c8;font-size:11px;margin-left:12px;">LOCAL PROTOTYPE ENVIRONMENT</span>
        </div>
        </div>
                """,
        unsafe_allow_html=True
    )
    # =========================
    # DETECTION COUNTS
    # =========================

    image_history = [
        x for x in history
        if not str(x.get("filename", "")).lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        )
    ]

    video_history = [
        x for x in history
        if str(x.get("filename", "")).lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        )
    ]

    image_count = len(image_history)
    video_count = len(video_history)
    accident_count = sum(1 for x in history if x.get("accident"))
    normal_count = len(history) - accident_count

    pending_reviews = sum(
        1 for x in history
        if x.get("review_flag", False)
    )

    st.markdown(
        '<div style="font-size:12px;color:#91a9c8;'
        'font-weight:800;letter-spacing:1.5px;'
        'margin:8px 0 12px;">DETECTION OVERVIEW</div>',
        unsafe_allow_html=True
    )

    metrics = [
        ("IMAGES ANALYSED", image_count, "▧", "#20d8f5"),
        ("VIDEOS ANALYSED", video_count, "▷", "#a78bfa"),
        ("POTENTIAL ACCIDENTS", accident_count, "⚠", "#ff5570"),
        ("PENDING REVIEWS", pending_reviews, "◷", "#f5bd55")
    ]

    cols = st.columns(4)

    for col, (label, value, symbol, accent) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:linear-gradient(145deg,#102943,#0a1a2e);
                    border:1px solid #244563;
                    border-top:3px solid {accent};
                    border-radius:13px;
                    padding:17px 16px;
                    min-height:125px;
                    box-shadow:0 8px 22px rgba(0,0,0,.12);
                ">
                    <div style="font-size:21px;color:{accent};">
                        {symbol}
                    </div>
                    <div style="
                        color:#f4f8ff;
                        font-size:29px;
                        font-weight:800;
                        margin:8px 0 3px;
                    ">{value}</div>
                    <div style="
                        color:#9bb2d0;
                        font-size:10px;
                        font-weight:700;
                        letter-spacing:.7px;
                    ">{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # =========================
    # INCIDENT MONITORING
    # =========================

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.65, 1])

    with left:
        st.markdown(
            """
            <div style="
                background:#0b1b31;
                border:1px solid #1d3554;
                border-radius:13px;
                padding:18px 20px;
                margin-bottom:12px;
            ">
                <div style="color:#f4f8ff;font-size:17px;
                    font-weight:750;">Incident monitoring</div>
                <div style="color:#91a9c8;font-size:11px;
                    margin-top:5px;">
                    Latest image and video analysis activity
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if history:
            df = pd.DataFrame([
                {
                    "Time": x.get("timestamp", "Not available"),
                    "Media": (
                        "Video"
                        if str(x.get("filename", "")).lower().endswith(
                            (".mp4", ".avi", ".mov", ".mkv")
                        )
                        else "Image"
                    ),
                    "File": x.get("filename", "Unknown"),
                    "Assessment": (
                        "Potential accident"
                        if x.get("accident")
                        else "No strong signal"
                    ),
                    "Accident score": (
                        f'{x.get("accident_probability", 0) * 100:.1f}%'
                    )
                }
                for x in history[-8:][::-1]
            ])

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        else:
                    st.markdown(
                        """
        <div style="background:linear-gradient(110deg,#102943,#0b1b31);border:1px solid #244563;border-radius:12px;padding:20px 22px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
        <div style="color:#20d8f5;font-size:10px;font-weight:800;letter-spacing:1.5px;">LIVE MONITORING</div>
        <div style="color:#f4f8ff;font-size:17px;font-weight:750;margin-top:7px;">Awaiting incoming analysis</div>
        <div style="color:#91a9c8;font-size:12px;margin-top:5px;">No incidents have been recorded in this session.</div>
        </div>
        <div style="background:#17334d;border:1px solid #31516f;border-radius:10px;padding:12px 16px;text-align:center;">
        <div style="color:#20d8f5;font-size:23px;font-weight:800;">0</div>
        <div style="color:#91a9c8;font-size:9px;font-weight:700;">ACTIVE CASES</div>
        </div>
        </div>
        <div style="height:1px;background:#244563;margin:17px 0;"></div>
        <div style="color:#a9c0de;font-size:11px;">NEXT ACTION</div>
        <div style="color:#f4f8ff;font-size:13px;font-weight:650;margin-top:5px;">Submit road footage through Live Detection to begin an analysis.</div>
        </div>
                        """,
                unsafe_allow_html=True
            )

    # =========================
        # ROADSAFE HERO VISUAL
        # =========================

        HERO_IMAGE = Path(__file__).parent / "roadsafe_hero_visual.jpeg"

        if HERO_IMAGE.exists():
            st.image(
                str(HERO_IMAGE),
                use_container_width=True
            )
        else:
            st.warning("RoadSafe hero image not found.")

    # =========================
    # RIGHT SIDE STATUS PANELS
    # =========================

    with right:

        st.markdown(
            """
            <div style="
                background:#0b1b31;
                border:1px solid #1d3554;
                border-radius:13px;
                padding:18px;
                margin-bottom:12px;
            ">
                <div style="color:#f4f8ff;font-size:16px;
                    font-weight:750;">Detection summary</div>
                <div style="color:#91a9c8;font-size:11px;
                    margin-top:5px;">Current session results</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric("No strong accident signal", normal_count)

        st.metric("Potential incidents", accident_count)

        st.markdown(
            f"""
            <div style="
                background:#102943;
                border:1px solid #244563;
                border-radius:11px;
                padding:15px;
                margin-top:12px;
            ">
                <div style="color:#91a9c8;font-size:10px;
                    font-weight:800;letter-spacing:1px;">
                    MODEL PERFORMANCE
                </div>
                <div style="color:#20d8f5;font-size:31px;
                    font-weight:800;margin-top:5px;">
                    77.01%
                </div>
                <div style="color:#a9c0de;font-size:11px;">
                    Previously recorded test accuracy
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="
                background:#102943;
                border:1px solid #244563;
                border-radius:11px;
                padding:15px;
                margin-top:12px;
            ">
                <div style="color:#20d8f5;font-weight:700;
                    font-size:12px;">RESPONSE NETWORK</div>
                <div style="color:#dce9ff;font-size:12px;
                    margin-top:8px;">
                    GPS and hospital discovery
                </div>
                <div style="color:#91a9c8;font-size:11px;
                    margin-top:5px;">
                    Open Response Network to view location-based
                    hospital recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =========================
    # FOOTER / DISCLAIMER
    # =========================

    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="
            background:rgba(16,37,62,.75);
            border:1px solid #294563;
            border-radius:9px;
            padding:12px 15px;
            color:#a9c0de;
            font-size:11px;
            line-height:1.6;
        ">
            <b style="color:#20d8f5;">PROTOTYPE NOTICE</b><br>
            The 77.01% figure is a previously recorded test result,
            not a live accuracy measurement. Video scores represent
            averaged sampled-frame predictions. Hospital notifications
            and emergency dispatch are simulated and are not connected
            to real emergency services.
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "Live Detection":

    # Intelligence module header
    st.markdown("""
        <div class="hero-card">
            <div class="hero-eyebrow">ROADSAFE AI / INTELLIGENCE MODULE</div>
            <h1>Live Incident Detection</h1>
            <p>Analyze road footage, identify potential accident signals,
            and initiate the incident review workflow.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="
            display:flex;
            align-items:center;
            gap:10px;
            padding:12px 16px;
            margin:16px 0 22px 0;
            border:1px solid rgba(0,220,180,0.25);
            border-radius:12px;
            background:rgba(0,220,180,0.06);
            color:#9debdc;
            font-size:13px;">
            <span style="color:#00d6a0;">●</span>
            AI ENGINE READY &nbsp; | &nbsp; MobileNetV2 + YOLO
            &nbsp; | &nbsp; Operator review enabled
        </div>
    """, unsafe_allow_html=True)

    # Detection mode
    st.markdown("### 01 / Select detection mode")

    detection_mode = st.radio(
        "Detection mode",
        ["Image", "Video"],
        horizontal=True,
        key="roadsafe_detection_mode",
        label_visibility="collapsed"
    )

    image = None
    uploaded_file = None

    # Upload and preview panels
    upload_col, preview_col = st.columns(
        [1, 1.25],
        gap="large"
    )

    with upload_col:

        st.markdown("### 02 / Upload road footage")

        if detection_mode == "Image":

            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png"],
                key="roadsafe_image_upload"
            )

        else:

            uploaded_file = st.file_uploader(
                "Choose a video",
                type=["mp4", "avi", "mov"],
                key="roadsafe_video_upload"
            )

        st.caption(
            "Supported formats: JPG, JPEG, PNG"
            if detection_mode == "Image"
            else "Supported formats: MP4, AVI, MOV"
        )

        st.markdown("""
            <div style="
                margin-top:14px;
                padding:14px;
                border-radius:12px;
                background:rgba(255,255,255,0.035);
                border:1px solid rgba(255,255,255,0.08);">
                <div style="font-weight:600;">Analysis pipeline</div>
                <div style="font-size:12px;color:#9caabd;
                    line-height:2;margin-top:8px;">
                    ① Media input<br>
                    ② MobileNetV2 classification<br>
                    ③ YOLO visual analysis<br>
                    ④ Preliminary incident assessment<br>
                    ⑤ Operator review
                </div>
            </div>
        """, unsafe_allow_html=True)

    with preview_col:

        st.markdown("### 03 / Media preview")

        if uploaded_file is not None:

            if detection_mode == "Image":

                try:
                    image = Image.open(uploaded_file).convert("RGB")

                    st.image(
                        image,
                        caption=f"Input image · {uploaded_file.name}",
                        use_container_width=True
                    )

                except Exception as exc:
                    st.error(f"Could not open image: {exc}")

            else:

                st.video(uploaded_file)

        else:

            st.markdown("""
                <div style="
                    min-height:260px;
                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                    border:1px dashed rgba(100,180,220,0.35);
                    border-radius:16px;
                    background:rgba(30,60,85,0.15);
                    text-align:center;
                    padding:20px;">
                    <div style="font-size:42px;">🎥</div>
                    <div style="font-size:16px;font-weight:600;
                        margin-top:12px;color:#dce8f5;">
                        Awaiting media input
                    </div>
                    <div style="font-size:12px;color:#91a4b9;
                        margin-top:6px;">
                        Upload road footage to preview it here
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    analyze = st.button(
        "⚡ ANALYZE ROAD FOOTAGE",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None
    )

    if analyze and uploaded_file is not None:

        if not MODEL_PATH.exists():
            st.error(f"Model not found: {MODEL_PATH.name}")

        else:
            try:
                with st.spinner("MobileNetV2 and YOLO are analyzing..."):

                    model = load_model()
                    visual_model = load_visual_model()

                    visual_observations = []

                    input_shape = model.input_shape
                    height = int(input_shape[1])
                    width = int(input_shape[2])

                    accident_scores = []
                    raw_outputs = []

                    if detection_mode == "Image":

                        image = Image.open(uploaded_file).convert("RGB")

                        # YOLO visual analysis
                        image_array = np.array(image)
                        bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                        visual_result = analyze_frame(visual_model, bgr_image)
                        visual_observations.append(visual_result)

                        resized = image.resize((width, height))

                        arr = np.expand_dims(
                            np.asarray(resized, dtype=np.float32),
                            axis=0
                        )

                        prediction = np.asarray(
                            model.predict(arr, verbose=0)
                        ).reshape(-1)

                        if prediction.size != 1:
                            raise ValueError(
                                "Expected a single sigmoid output."
                            )

                        normal_probability = float(
                            np.clip(prediction[0], 0, 1)
                        )

                        accident_probability = 1.0 - normal_probability

                        accident_scores.append(accident_probability)
                        raw_outputs.append(prediction.tolist())

                        filename = uploaded_file.name
                    else:
                        # Save uploaded video temporarily for OpenCV
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        ) as temp_video:

                            temp_video.write(uploaded_file.getvalue())
                            temp_path = temp_video.name

                        cap = cv2.VideoCapture(temp_path)

                        if not cap.isOpened():
                            raise ValueError("Could not open uploaded video.")

                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(
                            cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        )

                        # Analyze approximately one frame per second
                        frame_interval = max(1, int(fps)) if fps > 0 else 25

                        progress = st.progress(0)
                        status = st.empty()

                        frame_number = 0
                        analyzed_frames = 0
                        last_frame = None

                        try:
                            while True:
                                success, frame = cap.read()

                                if not success:
                                    break

                                if frame_number % frame_interval == 0:

                                    # Analyze the sampled frame using YOLO
                                    visual_result = analyze_frame(
                                        visual_model,
                                        frame
                                    )

                                    visual_observations.append(visual_result)

                                    rgb_frame = cv2.cvtColor(
                                        frame,
                                        cv2.COLOR_BGR2RGB
                                    )

                                    pil_image = Image.fromarray(rgb_frame)
                                    resized = pil_image.resize((width, height))

                                    arr = np.expand_dims(
                                        np.asarray(
                                            resized,
                                            dtype=np.float32
                                        ),
                                        axis=0
                                    )

                                    prediction = np.asarray(
                                        model.predict(arr, verbose=0)
                                    ).reshape(-1)

                                    if prediction.size != 1:
                                        raise ValueError(
                                            "Expected a single sigmoid output."
                                        )

                                    normal_probability = float(
                                        np.clip(prediction[0], 0, 1)
                                    )

                                    accident_probability = (
                                        1.0 - normal_probability
                                    )

                                    accident_scores.append(
                                        accident_probability
                                    )

                                    raw_outputs.append(prediction.tolist())

                                    last_frame = rgb_frame
                                    analyzed_frames += 1

                                    if total_frames > 0:
                                        progress.progress(
                                            min(
                                                frame_number / total_frames,
                                                1.0
                                            )
                                        )

                                    status.write(
                                        f"Analyzed frames: {analyzed_frames}"
                                    )

                                frame_number += 1

                        finally:
                            cap.release()
                            os.unlink(temp_path)

                        if not accident_scores:
                            raise ValueError(
                                "No frames could be analyzed."
                            )

                        st.image(
                            last_frame,
                            caption="Last analyzed video frame",
                            use_container_width=True
                        )

                        progress.progress(1.0)

                        filename = uploaded_file.name

                    # Aggregate YOLO observations across analyzed frames
                    if visual_observations:

                        max_vehicles = max(
                            x["vehicle_count"] for x in visual_observations
                        )

                        max_people = max(
                            x["person_count"] for x in visual_observations
                        )

                        all_vehicles = []

                        for observation in visual_observations:
                            for vehicle in observation["vehicles"]:
                                if vehicle["type"] not in all_vehicles:
                                    all_vehicles.append(vehicle["type"])

                        visual_summary = {
                            "vehicle_count": max_vehicles,
                            "person_count": max_people,
                            "vehicle_types": all_vehicles
                        }

                        visual_activity = preliminary_assessment(
                            visual_summary
                        )

                    else:
                        visual_summary = {
                            "vehicle_count": 0,
                            "person_count": 0,
                            "vehicle_types": []
                        }

                        visual_activity = "Insufficient visual evidence"

                    # Aggregate image/video prediction
                    accident_probability = float(
                        np.mean(accident_scores)
                    )

                    normal_probability = 1.0 - accident_probability

                    # Preliminary incident assessment
                    assessment = assess_incident(
                        accident_probability,
                        visual_activity
                    )

                    result = {
                        "accident": accident_probability >= 0.5,
                        "accident_probability": accident_probability,
                        "normal_probability": normal_probability,
                        "raw_output": raw_outputs,
                        "input_size": f"{width}x{height}",
                        "timestamp": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "filename": filename,
                        "detection_mode": detection_mode,
                        "frames_analyzed": len(accident_scores),
                        "vehicle_count": visual_summary["vehicle_count"],
                        "person_count": visual_summary["person_count"],
                        "vehicle_types": visual_summary["vehicle_types"],
                        "visual_activity": visual_activity,
                        "incident_priority": assessment["priority"],
                        "review_flag": assessment["review_flag"],
                        "review_reason": assessment["review_reason"],
                        "assessment_basis": assessment["assessment_basis"],
                        "requires_operator_review": assessment["requires_operator_review"],
                    }

                    st.session_state.roadsafe_last_result = result
                    st.session_state.roadsafe_history.append(result.copy())

                    # Create RescueChain response case
                    rescue_case = create_rescue_case(result)

                    if "rescue_cases" not in st.session_state:
                        st.session_state.rescue_cases = []

                    st.session_state.rescue_cases.append(rescue_case)

                    save_rescue_case(rescue_case)

                    # Save incident permanently in SQLite
                    save_data = result.copy()

                    save_data["media_type"] = (
                        "video" if str(result.get("filename", "")).lower().endswith(
                            (".mp4", ".avi", ".mov", ".mkv")
                        ) else "image"
                    )

                    save_incident(save_data)

            except Exception as exc:
                st.error("Detection failed.")
                st.exception(exc)

# Display latest result
    result = st.session_state.roadsafe_last_result

    if result:

        st.markdown("---")

        st.markdown("""
            <div style="margin:12px 0 20px 0;">
                <div style="font-size:11px;letter-spacing:2px;
                    color:#38c9df;font-weight:700;">
                    ROADSAFE AI / ANALYSIS REPORT
                </div>
                <div style="font-size:26px;font-weight:750;
                    color:#edf5ff;margin-top:5px;">
                    Detection results
                </div>
                <div style="font-size:13px;color:#91a4b9;
                    margin-top:5px;">
                    Model output and preliminary visual assessment
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Main result status
        if result["accident"]:

            st.markdown("""
                <div style="
                    padding:17px;
                    border-radius:14px;
                    border:1px solid rgba(255,90,90,0.35);
                    background:rgba(255,65,65,0.09);
                    margin-bottom:18px;">
                    <div style="color:#ff7777;font-size:11px;
                        letter-spacing:1.5px;font-weight:700;">
                        ⚠ POTENTIAL ACCIDENT SIGNAL
                    </div>
                    <div style="color:#f5f7fb;font-size:19px;
                        font-weight:700;margin-top:7px;">
                        Operator verification required
                    </div>
                    <div style="color:#b7c3d2;font-size:12px;
                        margin-top:5px;">
                        The model detected an accident-related signal.
                        This is not confirmation of an accident.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
                <div style="
                    padding:17px;
                    border-radius:14px;
                    border:1px solid rgba(0,210,160,0.3);
                    background:rgba(0,210,160,0.07);
                    margin-bottom:18px;">
                    <div style="color:#56dfb4;font-size:11px;
                        letter-spacing:1.5px;font-weight:700;">
                        ✓ NO STRONG ACCIDENT SIGNAL
                    </div>
                    <div style="color:#f5f7fb;font-size:19px;
                        font-weight:700;margin-top:7px;">
                        No strong signal detected
                    </div>
                    <div style="color:#b7c3d2;font-size:12px;
                        margin-top:5px;">
                        This result does not rule out an accident.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Prediction metrics
        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Accident probability",
                f'{result["accident_probability"] * 100:.1f}%'
            )
            st.progress(
                float(result["accident_probability"])
            )

        with metric2:
            st.metric(
                "Normal probability",
                f'{result["normal_probability"] * 100:.1f}%'
            )
            st.progress(
                float(result["normal_probability"])
            )

        # Analysis information
        st.markdown("### 01 / Analysis information")

        info1, info2, info3 = st.columns(3)

        info1.metric(
            "Frames analyzed",
            result.get("frames_analyzed", 1)
        )

        info2.metric(
            "Vehicles detected",
            result.get("vehicle_count", 0)
        )

        info3.metric(
            "People detected",
            result.get("person_count", 0)
        )

        with st.expander("View file and processing details"):
            st.write("**File:**", result.get("filename", "Not available"))
            st.write("**Detection mode:**", result.get("detection_mode", "Not available"))
            st.write("**Timestamp:**", result.get("timestamp", "Not available"))
            st.write("**Model input size:**", result.get("input_size", "Not available"))

        # YOLO analysis
        st.markdown("### 02 / YOLO visual analysis")

        yolo_col1, yolo_col2 = st.columns(2)

        with yolo_col1:
            st.markdown("**Detected vehicle types**")
            vehicle_types = result.get("vehicle_types", [])
            if vehicle_types:
                for vehicle in vehicle_types:
                    st.markdown(f"- {vehicle}")
            else:
                st.caption("No vehicle types detected.")

        with yolo_col2:
            st.markdown("**Visual activity**")
            st.info(
                result.get("visual_activity", "Not available")
            )

        st.caption(
            "YOLO identifies visible objects. Visual activity is not "
            "a validated measure of accident or injury severity."
        )

        # Preliminary assessment
        st.markdown("### 03 / Preliminary incident assessment")

        priority = result.get("incident_priority", "Not available")

        if priority == "High":
            st.error(f"Priority: {priority}")
        elif priority == "Moderate":
            st.warning(f"Priority: {priority}")
        elif priority == "Low":
            st.info(f"Priority: {priority}")
        else:
            st.info("Priority: Not available")

        if result.get("review_flag", False):
            st.warning(
                "⚠ Additional review required: "
                + result.get(
                    "review_reason",
                    "Visual activity requires further review."
                )
            )
        else:
            st.success("No additional visual activity review flag.")

        st.markdown("""
            <div style="
                padding:14px;
                border-radius:12px;
                background:rgba(255,180,60,0.07);
                border:1px solid rgba(255,180,60,0.22);
                color:#e8cda0;font-size:12px;margin-top:14px;">
                <b>Safety notice:</b> This is an experimental AI
                prioritization indicator, not a medical or injury-severity
                assessment. Every potential incident requires human review.
                A low priority does not rule out an accident.
            </div>
        """, unsafe_allow_html=True)

elif page == "Incident History":

    # Fetch permanent SQLite records
    db_records = get_all_incidents()

    # Page header
    st.markdown("""
        <div class="hero-card">
            <div class="hero-eyebrow">ROADSAFE AI / RECORDS MANAGEMENT</div>
            <h1>Incident History</h1>
            <p>Review, search and track previously analyzed road incidents.</p>
        </div>
    """, unsafe_allow_html=True)

    if db_records:

        # Convert database records into dictionaries
        all_rows = []

        for item in db_records:

            (
                incident_id,
                timestamp,
                filename,
                media_type,
                accident_probability,
                normal_probability,
                accident_detected,
                location,
                vehicle_count,
                person_count,
                vehicle_types,
                visual_activity,
                incident_priority,
                review_flag,
                review_reason
            ) = item

            all_rows.append({
                "ID": incident_id,
                "Time": timestamp,
                "Media": str(media_type or "image").title(),
                "Filename": filename or "Unknown",
                "Result": (
                    "Potential accident"
                    if accident_detected
                    else "No strong accident signal"
                ),
                "Accident score": float(accident_probability or 0),
                "Normal score": float(normal_probability or 0),
                "Vehicles": vehicle_count or 0,
                "People": person_count or 0,
                "Vehicle types": vehicle_types or "None detected",
                "Visual activity": visual_activity or "Not available",
                "Incident priority": incident_priority or "Not assessed",
                "Review flag": bool(review_flag),
                "Review reason": review_reason or "None",
                "Location": location or "Not available"
            })

        # Full database metrics (not affected by filters)
        total_incidents = len(all_rows)

        potential_accidents = sum(
            row["Result"] == "Potential accident"
            for row in all_rows
        )

        normal_detections = total_incidents - potential_accidents

        pending_reviews = sum(
            row["Review flag"] for row in all_rows
        )

        st.markdown("### 01 / Incident overview")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Total incidents", total_incidents)
        m2.metric("Potential accidents", potential_accidents)
        m3.metric("Normal detections", normal_detections)
        m4.metric("Review flagged", pending_reviews)

        st.markdown("---")

        # Search and filters
        st.markdown("### 02 / Search and filters")

        search_col, result_col, media_col = st.columns(
            [1.5, 1, 1]
        )

        with search_col:
            search = st.text_input(
                "Search records",
                placeholder="Search filename or incident ID...",
                key="history_search"
            )

        with result_col:
            kind = st.selectbox(
                "Detection result",
                [
                    "All results",
                    "Potential accident",
                    "No strong accident signal"
                ],
                key="history_result_filter"
            )

        with media_col:
            media_filter = st.selectbox(
                "Media type",
                ["All media", "Image", "Video"],
                key="history_media_filter"
            )

        # Additional priority filter
        priority_filter = st.selectbox(
            "Incident priority",
            [
                "All priorities",
                "High",
                "Moderate",
                "Low",
                "Not assessed"
            ],
            key="history_priority_filter"
        )

        # Apply filters
        filtered_rows = []

        for row in all_rows:

            search_text = search.strip().lower()

            if search_text and (
                search_text not in str(row["Filename"]).lower()
                and search_text not in str(row["ID"]).lower()
            ):
                continue

            if kind != "All results" and row["Result"] != kind:
                continue

            if (
                media_filter != "All media"
                and row["Media"].lower() != media_filter.lower()
            ):
                continue

            if (
                priority_filter != "All priorities"
                and row["Incident priority"] != priority_filter
            ):
                continue

            filtered_rows.append(row)

        st.markdown("---")

        # Incident records
        st.markdown("### 03 / Incident command log")

        st.caption(
            f"Showing {len(filtered_rows)} of {total_incidents} permanent records."
        )

        if filtered_rows:

            for row in filtered_rows:

                accident_percent = row["Accident score"] * 100

                with st.container(border=True):

                    header_col, badge_col = st.columns(
                        [2, 1]
                    )

                    with header_col:
                        st.markdown(
                            f"**Incident #{row['ID']}**"
                        )
                        st.caption(
                            f"{row['Filename']}  ·  {row['Time']}"
                        )

                    with badge_col:

                        if row["Result"] == "Potential accident":
                            st.error("⚠ Potential accident")
                        else:
                            st.success("✓ Normal signal")

                    score_col, priority_col, review_col = st.columns(3)

                    score_col.metric(
                        "Accident probability",
                        f"{accident_percent:.1f}%"
                    )

                    priority_col.metric(
                        "Priority",
                        row["Incident priority"]
                    )

                    review_col.metric(
                        "Review flagged",
                        "Yes" if row["Review flag"] else "No"
                    )

                    with st.expander("View complete incident details"):

                        detail1, detail2 = st.columns(2)

                        with detail1:
                            st.markdown("**Detection information**")
                            st.write("Media type:", row["Media"])
                            st.write("Filename:", row["Filename"])
                            st.write(
                                "Normal probability:",
                                f"{row['Normal score'] * 100:.1f}%"
                            )
                            st.write(
                                "Accident probability:",
                                f"{accident_percent:.1f}%"
                            )
                            st.write("Location:", row["Location"])

                        with detail2:
                            st.markdown("**Visual analysis**")
                            st.write("Vehicles:", row["Vehicles"])
                            st.write("People:", row["People"])
                            st.write(
                                "Vehicle types:",
                                row["Vehicle types"]
                            )
                            st.write(
                                "Visual activity:",
                                row["Visual activity"]
                            )

                        st.markdown("**Review information**")
                        st.write(
                            "Review reason:",
                            row["Review reason"]
                        )

                        st.caption(
                            "Historical model output. This record is not "
                            "a confirmed accident or medical assessment."
                        )

        else:
            st.info("No incidents match your selected filters.")

    else:

        st.markdown("""
            <div style="
                padding:35px 20px;
                text-align:center;
                border:1px dashed rgba(100,180,220,0.3);
                border-radius:16px;
                background:rgba(30,60,85,0.12);">
                <div style="font-size:38px;">📂</div>
                <div style="font-size:19px;font-weight:700;
                    color:#e5edf7;margin-top:10px;">
                    No incident records yet
                </div>
                <div style="font-size:13px;color:#91a4b9;
                    margin-top:7px;">
                    Analyze an image or video in Live Detection
                    to create your first permanent record.
                </div>
            </div>
        """, unsafe_allow_html=True)

elif page == "RescueChain":

    rescue_cases = get_all_rescue_cases()

    # Command centre header
    st.markdown("""
        <div class="hero-card">
            <div class="hero-eyebrow">ROADSAFE AI / EMERGENCY OPERATIONS</div>
            <h1>RescueChain</h1>
            <p>Incident verification, response coordination and hospital acknowledgement.</p>
        </div>
    """, unsafe_allow_html=True)

    # Workflow summary
    total_cases = len(rescue_cases)

    awaiting_review = sum(
        not case["operator_verified"] for case in rescue_cases
    )

    response_initiated = sum(
        case["response_status"] == "Response initiated"
        for case in rescue_cases
    )

    acknowledged = sum(
        case["response_status"] == "Response acknowledged"
        for case in rescue_cases
    )

    assistance_confirmed = sum(
        case["response_status"] == "Assistance confirmed"
        for case in rescue_cases
    )

    st.markdown("### 01 / Response overview")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Total cases", total_cases)
    m2.metric("Awaiting verification", awaiting_review)
    m3.metric("Response initiated", response_initiated)
    m4.metric("Assistance confirmed", assistance_confirmed)

    st.markdown("---")

    # Workflow stages
    st.markdown("### 02 / Emergency response workflow")

    stages = [
        ("01", "Incident detected"),
        ("02", "Operator verification"),
        ("03", "Response initiated"),
        ("04", "Hospital acknowledged"),
        ("05", "Assistance confirmed")
    ]

    stage_cols = st.columns(5)

    for col, (number, label) in zip(stage_cols, stages):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:rgba(40,65,90,0.22);
                    border:1px solid rgba(100,180,220,0.18);
                    border-radius:12px;
                    padding:12px 7px;
                    text-align:center;
                    min-height:100px;">
                    <div style="color:#45c9e5;
                        font-size:19px;font-weight:800;">
                        {number}
                    </div>
                    <div style="font-size:11px;
                        color:#d5e1ed;margin-top:8px;">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("### 03 / Incident response cases")

    if not rescue_cases:

        st.info(
            "No incoming response cases yet. "
            "Analyze an image or video in Live Detection first."
        )

    else:

        st.caption(
            f"{total_cases} response case(s) retrieved from permanent SQLite storage."
        )

        for case in rescue_cases:

            case_id = case["case_id"]
            response_status = case["response_status"]

            if response_status == "Assistance confirmed":
                status_label = "ASSISTANCE CONFIRMED"
            elif response_status == "Response acknowledged":
                status_label = "HOSPITAL ACKNOWLEDGED"
            elif response_status == "Response initiated":
                status_label = "RESPONSE INITIATED"
            elif case["operator_verified"]:
                status_label = "OPERATOR VERIFIED"
            else:
                status_label = "AWAITING VERIFICATION"

            with st.container(border=True):

                # Case heading
                header_col, status_col = st.columns([2, 1])

                with header_col:
                    st.markdown(f"**{case_id}**")
                    st.caption(
                        f'{case.get("filename", "Unknown")} · '
                        f'{case.get("created_at", "Time unavailable")}'
                    )

                with status_col:
                    st.markdown(
                        f"""
                        <div style="
                            text-align:center;
                            padding:8px 5px;
                            border-radius:8px;
                            background:rgba(50,150,190,0.12);
                            border:1px solid rgba(80,190,220,0.25);
                            color:#78d9ef;
                            font-size:10px;
                            font-weight:700;">
                            {status_label}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Case metrics
                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Accident probability",
                    f'{case["accident_probability"] * 100:.1f}%'
                )

                c2.metric(
                    "Priority",
                    case["incident_priority"]
                )

                c3.metric(
                    "Operator verified",
                    "Yes" if case["operator_verified"] else "No"
                )

                st.progress(
                    min(
                        max(float(case["accident_probability"]), 0.0),
                        1.0
                    )
                )

                with st.expander("View case details"):

                    st.write(
                        "**Created:**",
                        case.get("created_at", "Not available")
                    )

                    st.write(
                        "**Review flag:**",
                        "Required" if case["review_flag"] else "Not flagged"
                    )

                    st.write(
                        "**Current case status:**",
                        case["status"]
                    )

                    st.write(
                        "**Response status:**",
                        case["response_status"]
                    )

                    st.write(
                        "**Hospital:**",
                        case.get("hospital_name", "Not assigned")
                    )

                    st.divider()

                    # Operator verification
                    st.markdown("#### 01 / Operator verification")

                    if not case["operator_verified"]:

                        st.warning(
                            "This case requires operator verification "
                            "before a response can be initiated."
                        )

                        if st.button(
                            "✓ Verify incident",
                            key=f"verify_{case_id}",
                            type="primary"
                        ):

                            case["operator_verified"] = True
                            case["status"] = "Verified by operator"

                            update_rescue_case(case)

                            st.success("Incident verified.")
                            st.rerun()

                    else:

                        st.success("Incident verified by operator.")

                    st.divider()

                    # Response initiation
                    st.markdown("#### 02 / Response coordination")

                    if case["operator_verified"]:

                        if response_status == "Not initiated":

                            st.info(
                                "The case is verified and ready for "
                                "simulated response initiation."
                            )

                            if st.button(
                                "⚡ Initiate simulated response",
                                key=f"response_{case_id}",
                                type="primary"
                            ):

                                success = initiate_response(case)

                                if success:
                                    update_rescue_case(case)
                                    st.success(
                                        "Simulated response initiated."
                                    )
                                    st.rerun()

                        else:

                            st.success(
                                f"Response status: {response_status}"
                            )

                            st.write(
                                "Response timestamp:",
                                case.get(
                                    "response_timestamp",
                                    "Not available"
                                )
                            )

                    st.divider()

                    # Hospital acknowledgement
                    st.markdown("#### 03 / Hospital acknowledgement")

                    if response_status == "Response initiated":

                        components.html(
                            """
                            <div style="font-family:Arial,sans-serif;">
                                <button id="alarmBtn"
                                    style="padding:11px 15px;
                                    background:#b91c1c;
                                    color:white;font-weight:bold;
                                    border:0;border-radius:8px;
                                    cursor:pointer;">
                                    🔔 Enable hospital alert sound
                                </button>
                                <p id="alarmStatus"
                                   style="color:#b8c7d8;font-size:12px;">
                                   Sound disabled. Enable manually if required.
                                </p>
                                <script>
                                const AudioContext =
                                    window.AudioContext ||
                                    window.webkitAudioContext;
                                let audioContext;
                                let alarmInterval;

                                function beep() {
                                    const oscillator =
                                        audioContext.createOscillator();
                                    const gain =
                                        audioContext.createGain();

                                    oscillator.connect(gain);
                                    gain.connect(audioContext.destination);
                                    oscillator.type = "square";
                                    oscillator.frequency.value = 1000;
                                    gain.gain.value = 0.20;
                                    oscillator.start();
                                    oscillator.stop(
                                        audioContext.currentTime + 0.35
                                    );
                                }

                                document.getElementById("alarmBtn")
                                    .addEventListener("click", async () => {
                                        if (!audioContext) {
                                            audioContext = new AudioContext();
                                        }
                                        await audioContext.resume();

                                        if (alarmInterval) {
                                            clearInterval(alarmInterval);
                                        }

                                        beep();
                                        alarmInterval = setInterval(beep, 1000);

                                        document.getElementById("alarmStatus")
                                            .innerText =
                                            "Hospital acknowledgement pending — alarm active in this panel.";
                                    });
                                </script>
                            </div>
                            """,
                            height=100
                        )

                        if not case.get("acknowledgement_pending", False):

                            if st.button(
                                "Acknowledge alert",
                                key=f"ack_{case_id}"
                            ):
                                case["acknowledgement_pending"] = True
                                update_rescue_case(case)
                                st.rerun()

                        else:

                            st.warning(
                                "Confirm that the hospital has received "
                                "the incident alert."
                            )

                            ack_col, cancel_col = st.columns(2)

                            with ack_col:
                                if st.button(
                                    "Confirm acknowledgement",
                                    key=f"confirm_ack_{case_id}"
                                ):

                                    success = acknowledge_response(case)

                                    if success:
                                        case["acknowledgement_pending"] = False
                                        update_rescue_case(case)
                                        st.success(
                                            "Hospital acknowledgement recorded."
                                        )
                                        st.rerun()

                            with cancel_col:
                                if st.button(
                                    "Cancel acknowledgement",
                                    key=f"cancel_ack_{case_id}"
                                ):
                                    case["acknowledgement_pending"] = False
                                    update_rescue_case(case)
                                    st.rerun()

                    elif response_status == "Response acknowledged":

                        st.success(
                            "Hospital coordinator has acknowledged the alert."
                        )

                        st.write(
                            "Acknowledgement timestamp:",
                            case.get(
                                "acknowledgement_timestamp",
                                "Not available"
                            )
                        )

                    elif response_status == "Assistance confirmed":

                        st.success(
                            "Hospital assistance confirmed (simulation)."
                        )

                        st.write(
                            "Assistance timestamp:",
                            case.get(
                                "assistance_timestamp",
                                "Not available"
                            )
                        )

                    # Hospital assistance confirmation
                    if response_status == "Response acknowledged":

                        st.divider()
                        st.markdown("#### 04 / Hospital assistance")

                        if st.button(
                            "✓ Confirm hospital assistance",
                            key=f"assist_{case_id}"
                        ):

                            success = confirm_assistance(case)

                            if success:
                                update_rescue_case(case)
                                st.success(
                                    "Hospital assistance confirmed (simulation)."
                                )
                                st.rerun()

    st.markdown("---")

    st.caption(
        "DEMO ENVIRONMENT · Response initiation, hospital acknowledgement "
        "and assistance confirmation are simulated. No actual emergency "
        "services or hospitals are contacted."
    )

elif page == "Hospital Dashboard":

    hospital_cases = get_all_rescue_cases()

    incoming_cases = [
        case for case in hospital_cases
        if case.get("response_status") in [
            "Response initiated",
            "Response acknowledged",
            "Assistance confirmed"
        ]
    ]

    awaiting_ack = sum(
        1 for case in incoming_cases
        if case.get("response_status") == "Response initiated"
    )

    acknowledged = sum(
        1 for case in incoming_cases
        if case.get("response_status") == "Response acknowledged"
    )

    assistance_confirmed = sum(
        1 for case in incoming_cases
        if case.get("response_status") == "Assistance confirmed"
    )

    # HOSPITAL DASHBOARD HEADER
    st.markdown("""
        <div style="padding:28px 30px; border-radius:18px; background:linear-gradient(115deg,#102b3d,#123b50,#10283b); border:1px solid rgba(56,189,248,0.25); margin-bottom:22px;">
        <div style="color:#38bdf8;font-size:11px;font-weight:800;letter-spacing:3px;">ROADSAFE AI / MEDICAL RESPONSE UNIT</div>
        <div style="color:#f1f5f9;font-size:29px;font-weight:800;margin-top:9px;">Hospital Operations Dashboard</div>
        <div style="color:#a8bdca;font-size:13px;margin-top:7px;">Incoming incident alerts · Hospital acknowledgement · Assistance coordination</div>
        </div>
        """, unsafe_allow_html=True)
    
    # STATUS STRIP
    st.markdown("""
    <div style="
        padding:12px 16px;
        border-radius:10px;
        background:rgba(34,197,94,0.07);
        border:1px solid rgba(34,197,94,0.22);
        margin-bottom:20px;
        color:#86efac;
        font-size:12px;
        font-weight:600;
    ">
        ● HOSPITAL RESPONSE INTERFACE &nbsp; | &nbsp;
        LIVE DATABASE RECORDS
    </div>
    """, unsafe_allow_html=True)

    # METRICS
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Incoming Alerts", len(incoming_cases))

    with m2:
        st.metric("Awaiting Acknowledgement", awaiting_ack)

    with m3:
        st.metric("Acknowledged", acknowledged)

    with m4:
        st.metric("Assistance Confirmed", assistance_confirmed)

    st.markdown("---")

    # INCIDENT QUEUE
    st.markdown("""
    <div style="margin-bottom:14px;">
        <div style="
            color:#38bdf8;
            font-size:11px;
            font-weight:800;
            letter-spacing:2px;
        ">MEDICAL RESPONSE QUEUE</div>
        <div style="
            color:#f1f5f9;
            font-size:21px;
            font-weight:750;
            margin-top:5px;
        ">Incoming Incident Cases</div>
    </div>
    """, unsafe_allow_html=True)

    if not incoming_cases:

        st.markdown("""
        <div style="
            padding:35px 20px;
            text-align:center;
            border:1px dashed rgba(148,163,184,0.3);
            border-radius:15px;
            background:rgba(15,23,42,0.35);
        ">
            <div style="font-size:30px;">✚</div>
            <div style="
                color:#e2e8f0;
                font-size:17px;
                font-weight:700;
                margin-top:10px;
            ">No Incoming Hospital Alerts</div>
            <div style="
                color:#94a3b8;
                font-size:12px;
                margin-top:6px;
            ">New response cases will appear here when initiated through RescueChain.</div>
        </div>
        """, unsafe_allow_html=True)

    else:

        for case in incoming_cases:

            case_id = case.get("case_id", "Unknown")
            response_status = case.get("response_status", "Unknown")
            priority = case.get("incident_priority", "Not assessed")
            probability = float(case.get("accident_probability") or 0)
            probability = max(0.0, min(1.0, probability))

            if response_status == "Response initiated":
                status_color = "#fbbf24"
                status_label = "AWAITING ACKNOWLEDGEMENT"
                status_icon = "◷"

            elif response_status == "Response acknowledged":
                status_color = "#38bdf8"
                status_label = "ALERT ACKNOWLEDGED"
                status_icon = "✓"

            else:
                status_color = "#4ade80"
                status_label = "ASSISTANCE CONFIRMED"
                status_icon = "✓"

            with st.container(border=True):

                left, right = st.columns([3, 1])

                with left:
                    st.markdown(f"""
                    <div style="
                        color:#38bdf8;
                        font-size:10px;
                        font-weight:800;
                        letter-spacing:1.5px;
                    ">INCIDENT REFERENCE</div>

                    <div style="
                        color:#f1f5f9;
                        font-size:19px;
                        font-weight:800;
                        margin-top:5px;
                    ">{case_id}</div>

                    <div style="
                        color:{status_color};
                        font-size:10px;
                        font-weight:800;
                        letter-spacing:1px;
                        margin-top:9px;
                    ">{status_icon} &nbsp; {status_label}</div>
                    """, unsafe_allow_html=True)

                with right:
                    st.metric("Accident Probability", f"{probability * 100:.1f}%")

                st.progress(probability)

                st.markdown("")

                d1, d2, d3 = st.columns(3)

                with d1:
                    st.caption("INCIDENT PRIORITY")
                    st.write(f"**{priority}**")

                with d2:
                    st.caption("CASE STATUS")
                    st.write(f"**{case.get('status', 'Unknown')}**")

                with d3:
                    st.caption("CREATED AT")
                    st.write(f"**{case.get('created_at', 'Not available')}**")

                with st.expander("View Incident Details"):

                    st.write("**Incident file:**", case.get("filename", "Unknown"))
                    st.write("**Case ID:**", case_id)
                    st.write("**Response status:**", response_status)
                    st.write("**Operator verified:**", "Yes" if case.get("operator_verified") else "No")

                    if case.get("acknowledgement_timestamp"):
                        st.write(
                            "**Acknowledged at:**",
                            case["acknowledgement_timestamp"]
                        )

                    if case.get("assistance_timestamp"):
                        st.write(
                            "**Assistance confirmed at:**",
                            case["assistance_timestamp"]
                        )

                # HOSPITAL ACTIONS
                if response_status == "Response initiated":

                    if st.button(
                        "✓  Acknowledge Incident Alert",
                        key=f'hospital_ack_{case_id}',
                        use_container_width=True
                    ):

                        success = acknowledge_response(case)

                        if success:
                            update_rescue_case(case)
                            st.success("Incident alert acknowledged successfully.")
                            st.rerun()

                elif response_status == "Response acknowledged":

                    if st.button(
                        "✚  Confirm Hospital Assistance",
                        key=f'hospital_assist_{case_id}',
                        use_container_width=True
                    ):

                        success = confirm_assistance(case)

                        if success:
                            update_rescue_case(case)
                            st.success("Hospital assistance confirmed (simulation).")
                            st.rerun()

                else:
                    st.success("Hospital assistance confirmation recorded.")

    st.markdown("---")

    st.caption(
        "RoadSafe AI · Hospital response workflow is a demonstration. "
        "No real hospital, ambulance or emergency service is contacted."
    )

elif page == "Response Network":

    # GET DEVICE LOCATION
    if not GEOLOCATION_AVAILABLE:
        latitude = longitude = accuracy = None
    else:
        try:
            location_data = streamlit_geolocation()

            latitude = location_data.get("latitude") if isinstance(location_data, dict) else None
            longitude = location_data.get("longitude") if isinstance(location_data, dict) else None
            accuracy = location_data.get("accuracy") if isinstance(location_data, dict) else None

            if latitude is not None and longitude is not None:
                st.session_state["roadsafe_gps_latitude"] = float(latitude)
                st.session_state["roadsafe_gps_longitude"] = float(longitude)

                if accuracy is not None:
                    st.session_state["roadsafe_gps_accuracy"] = float(accuracy)

        except Exception as exc:
            st.error(f"Could not read browser location: {exc}")
            latitude = longitude = accuracy = None

    latitude = st.session_state.get("roadsafe_gps_latitude", latitude)
    longitude = st.session_state.get("roadsafe_gps_longitude", longitude)
    accuracy = st.session_state.get("roadsafe_gps_accuracy", accuracy)

    gps_connected = latitude is not None and longitude is not None

    # RESPONSE NETWORK HEADER
    st.markdown("""
<div style="padding:28px 30px;border-radius:18px;background:linear-gradient(115deg,#102b3d,#123b50,#10283b);border:1px solid rgba(56,189,248,0.25);margin-bottom:22px;">
<div style="color:#38bdf8;font-size:11px;font-weight:800;letter-spacing:3px;">ROADSAFE AI / GEOSPATIAL RESPONSE SYSTEM</div>
<div style="color:#f1f5f9;font-size:29px;font-weight:800;margin-top:9px;">Response Network</div>
<div style="color:#a8bdca;font-size:13px;margin-top:7px;">Device geolocation · Nearby medical facilities · Emergency response coordination</div>
</div>
""", unsafe_allow_html=True)

    # SAFETY NOTICE
    st.markdown("""
<div style="padding:12px 16px;border-radius:10px;background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.25);color:#fcd38d;font-size:12px;margin-bottom:20px;">
⚠ DEMONSTRATION MODE — GPS and map searches are functional, but hospital alerts and emergency dispatch are not connected to real services.
</div>
""", unsafe_allow_html=True)

    # NETWORK STATUS
    st.markdown("#### Network Status")

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "GPS Connection",
            "Connected" if gps_connected else "Permission needed"
        )

    with b:
        st.metric("Hospital Alerts", "Demo only")

    with c:
        st.metric("Emergency Dispatch", "Not sent")

    st.markdown("---")

    # LOCATION PANEL
    st.markdown("#### Device Location")

    if not GEOLOCATION_AVAILABLE:
        st.warning("GPS component is not installed.")
        st.code("python -m pip install streamlit-geolocation", language="bash")
        st.caption("Install the package, restart Streamlit, and reopen this page.")

    if gps_connected:

        st.success("Device coordinates received.")

        x, y, z = st.columns(3)

        with x:
            st.metric("Latitude", f"{latitude:.6f}")

        with y:
            st.metric("Longitude", f"{longitude:.6f}")

        with z:
            st.metric(
                "Location Accuracy",
                f"±{accuracy:.0f} m" if accuracy is not None else "Not provided"
            )

        maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"

        st.link_button(
            "⌖  Open Current Location in Google Maps",
            maps_url,
            use_container_width=True
        )

        st.caption(
            "Coordinates are stored only in this Streamlit session. "
            "They are not automatically transmitted to hospitals or emergency services."
        )

    else:
        st.info(
            "Location is not available. Allow browser location permission, "
            "enable device Location, and use HTTPS or localhost."
        )

    st.markdown("---")

    # NEARBY HOSPITALS
    st.markdown("#### Nearby Medical Facilities")

    if gps_connected:

        st.markdown("""
        <div style="padding:16px;border-radius:12px;background:#0b1b31;border:1px solid #1d3554;margin-bottom:12px;">
        <div style="color:#e2e8f0;font-weight:700;font-size:15px;">Find Hospitals Around Your Location</div>
        <div style="color:#91a9c8;font-size:12px;margin-top:6px;">Open a Google Maps search using your current device coordinates.</div>
        </div>
        """, unsafe_allow_html=True)

        nearby_url = (
            f"https://www.google.com/maps/search/hospitals/"
            f"@{latitude},{longitude},14z"
        )

        st.link_button(
            "✚  Search Nearby Hospitals",
            nearby_url,
            use_container_width=True
        )

        st.caption(
            "Google Maps provides the search results. Hospital availability, "
            "distance and capacity are not independently verified by RoadSafe AI."
        )

    else:
        st.info("Allow GPS access to search for nearby hospitals.")

    # RESPONSE WORKFLOW SIMULATOR
    st.markdown("#### Emergency Response & Voice Call Simulator")

    st.caption(
        "A browser-based simulated emergency call for testing the RoadSafe-AI response workflow."
    )

    last_result = st.session_state.get("roadsafe_last_result")
    hospital_case_id = st.session_state.get("roadsafe_hospital_case_id")

    hospital_case = None

    if hospital_case_id:
        all_cases = get_all_rescue_cases()

        hospital_case = next(
            (
                case for case in all_cases
                if case.get("case_id") == hospital_case_id
            ),
            None
        )
    incident_detected = bool(last_result and last_result.get("accident"))

    if incident_detected:
        st.error("Latest model result: Potential accident")
    elif last_result:
        st.success("Latest model result: No strong accident signal")
    else:
        st.info("Latest model result: No analysis available")

    if "roadsafe_call_status" not in st.session_state:
        st.session_state["roadsafe_call_status"] = "Not initiated"

    if "roadsafe_call_time" not in st.session_state:
        st.session_state["roadsafe_call_time"] = None

    with st.container(border=True):

        st.markdown("### Incoming Emergency Call")
        st.caption("ROADSAFE AI · AUTOMATED VOICE RESPONSE")

        if st.session_state["roadsafe_call_status"] == "Not initiated":

            st.info("Ready to initiate a simulated emergency call.")

        if st.button(
            "☎  Initiate Test Call",
            type="primary",
            use_container_width=True
        ):
            call_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Prepare incident data
            incident_data = last_result.copy() if last_result else {
                "filename": "Manual Test Incident",
                "timestamp": call_time,
                "accident_probability": 0.0,
                "incident_priority": "Test Incident",
                "review_flag": False
            }

            # Create a RescueChain case
            hospital_case = create_rescue_case(incident_data)

            # Mark this as a simulated, operator-verified test dispatch
            hospital_case["hospital_name"] = "RoadSafe-AI Demo Hospital"
            hospital_case["operator_verified"] = True
            hospital_case["status"] = "Response in progress"
            hospital_case["response_status"] = "Response initiated"
            hospital_case["response_timestamp"] = call_time
            hospital_case["acknowledgement_pending"] = True

            # Save to SQLite
            save_rescue_case(hospital_case)

            # Update session state
            if "rescue_cases" not in st.session_state:
                st.session_state.rescue_cases = []

            st.session_state.rescue_cases.append(hospital_case)

            st.session_state["roadsafe_call_status"] = "Call initiated"
            st.session_state["roadsafe_call_time"] = call_time
            st.session_state["roadsafe_hospital_case_id"] = hospital_case["case_id"]

            st.rerun()

        elif st.session_state["roadsafe_call_status"] == "Call initiated":

            st.warning("☎  SIMULATED INCOMING CALL — ROADSAFE AI")

            incident_message = (
                "Hello. This is an automated RoadSafe AI emergency test call. "
                "A potential road accident has been detected. "
                "Please review the incident and acknowledge this call. "
                "This is a demonstration only. No real emergency services have been contacted."
            )

            safe_message = incident_message.replace("\\", "\\\\").replace("'", "\\'")

            st.components.v1.html(
                f"""
                <script>
                if (!window.parent.roadsafeCallSpoken) {{
                    window.parent.roadsafeCallSpoken = true;
                    const message = new SpeechSynthesisUtterance('{safe_message}');
                    message.rate = 0.9;
                    message.pitch = 1;
                    window.speechSynthesis.cancel();
                    window.speechSynthesis.speak(message);
                }}
                </script>
                <div style="font-family:Arial;color:#38bdf8;padding:8px;">
                    <b>Automated voice message is playing in your browser.</b>
                </div>
                """,
                height=55
            )

            st.write("**Call initiated at:**", st.session_state["roadsafe_call_time"])

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✓ Acknowledge Call", use_container_width=True):
                    st.session_state["roadsafe_call_status"] = "Acknowledged"
                    st.session_state["roadsafe_call_response_time"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    st.rerun()

            with col2:
                if st.button("✕ Reject Call", use_container_width=True):
                    st.session_state["roadsafe_call_status"] = "Rejected"
                    st.session_state["roadsafe_call_response_time"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    st.rerun()

        else:

            if st.session_state["roadsafe_call_status"] == "Acknowledged":
                st.success("Call acknowledged successfully.")
            else:
                st.warning("Simulated call was rejected.")

            st.write("**Call status:**", st.session_state["roadsafe_call_status"])
            st.write("**Initiated:**", st.session_state["roadsafe_call_time"])

            if hospital_case:
                st.write("**Incident ID:**", hospital_case["case_id"])
                st.write("**Hospital:**", hospital_case.get("hospital_name", "Not assigned"))
                st.write("**Hospital response:**", hospital_case.get("response_status", "Pending"))

                st.write(
                    "**Hospital acknowledgement:**",
                    hospital_case.get("acknowledgement_timestamp") or "Awaiting acknowledgement"
                )

                st.write(
                    "**Assistance confirmation:**",
                    hospital_case.get("assistance_timestamp") or "Awaiting confirmation"
                )
            else:
                st.info("No linked hospital incident found.")

            st.write(
                "**Your call response time:**",
                st.session_state.get("roadsafe_call_response_time", "Not available")
            )

            st.markdown("---")
            st.markdown("### Incident Response Summary")

            st.write(
                "**Incident:**",
                "Potential accident" if incident_detected else "No strong accident signal"
            )
            st.write("**GPS status:**", "Connected" if gps_connected else "Unavailable")
            st.write("**Call outcome:**", st.session_state["roadsafe_call_status"])

            st.success("Response summary generated.")

            if st.button("Reset Test Call", use_container_width=True):
                st.session_state["roadsafe_call_status"] = "Not initiated"
                st.session_state["roadsafe_call_time"] = None
                st.session_state.pop("roadsafe_call_response_time", None)
                st.components.v1.html(
                    "<script>window.parent.roadsafeCallSpoken = false;</script>",
                    height=0
                )
                st.rerun()

elif page == "Model Insights":
    st.subheader("Model Insights")
    st.caption(
        "Recorded MobileNetV2 evaluation results from a previous test run."
    )

    # OVERALL PERFORMANCE
    st.markdown("### Overall Performance")

    a, b, c = st.columns(3)

    a.metric("Test Accuracy", "77.01%")
    b.metric("Accident Recall", "90.95%")
    c.metric("Accident Precision", "72.93%")

    st.markdown("---")

    # CLASS-WISE PERFORMANCE
    st.markdown("### Class-wise Performance")

    metrics_df = pd.DataFrame(
        {
            "Precision": [0.7293, 0.8528],
            "Recall": [0.9095, 0.6082]
        },
        index=["Accident", "Normal"]
    )

    st.dataframe(
        metrics_df.style.format("{:.2%}"),
        use_container_width=True
    )

    st.markdown("### Performance Comparison")
    st.bar_chart(metrics_df, use_container_width=True)

    st.markdown("---")

    # CONFUSION MATRIX
    st.markdown("### Confusion Matrix")

    st.caption(
        "Rows represent actual classes; columns represent predicted classes."
    )

    cm = pd.DataFrame(
        [[1307, 130], [485, 753]],
        index=["Actual Accident", "Actual Normal"],
        columns=["Predicted Accident", "Predicted Normal"]
    )

    st.dataframe(cm, use_container_width=True)

    st.markdown("---")

    # MODEL INTERPRETATION
    st.markdown("### Model Interpretation")

    st.success(
        "The model correctly identified approximately 91% of accident images "
        "in the recorded test evaluation."
    )

    st.warning(
        "The model also incorrectly classified 485 normal images as accidents. "
        "Further validation and improvement are required before real-world use."
    )

    st.info(
        "These are previously recorded evaluation results, not live performance "
        "calculations. RoadSafe-AI is a research prototype and must not replace "
        "emergency services or professional assessment."
    )

st.markdown('<div class="footer">ROADSAFE AI · SUPER 30 PROJECT · MOBILE NETV2 ACCIDENT CLASSIFICATION · PROTOTYPE DEMONSTRATION</div>', unsafe_allow_html=True)
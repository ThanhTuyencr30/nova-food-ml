import base64
import html
import tempfile
from collections import Counter
import io
import hashlib
import re
import time
from pathlib import Path
from contextlib import contextmanager
from difflib import SequenceMatcher
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageEnhance, ImageOps
try:
    import zxingcpp
except ImportError:
    zxingcpp = None
try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:
    zbar_decode = None
try:
    import requests
except ImportError:
    requests = None
from src.nova.inference.predict import (
    predict_ingredients,
    predict_proba_ingredients,
)
# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="NOVA Food Classifier",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)
APP_NAME = "NOVA Food Classifier"
USER_AGENT = "NOVA-Food-Classifier/2.0 (educational project)"
SCAN_PAGE = "Quét mã"
SEARCH_PAGE = "Tìm sản phẩm"
PREDICT_PAGE = "Phân loại"
HISTORY_PAGE = "Lịch sử"
LEARN_PAGE = "Kiến thức NOVA"
MODEL_PAGE = "Mô hình"
NAV_ITEMS = [SCAN_PAGE, SEARCH_PAGE, PREDICT_PAGE, HISTORY_PAGE, LEARN_PAGE, MODEL_PAGE]
# ============================================================
# DỮ LIỆU NOVA
# ============================================================
NOVA_META = {
    1: {
        "title": "Chưa hoặc ít chế biến",
        "color": "#2E9B5B",
        "explain": (
            "Thực phẩm tự nhiên hoặc chỉ qua xử lý tối thiểu như rửa, sấy, "
            "xay, đông lạnh, không thêm chất phụ gia."
        ),
        "examples": "Rau củ, trái cây, thịt cá tươi, trứng, sữa tươi, gạo, đậu.",
    },
    2: {
        "title": "Nguyên liệu chế biến",
        "color": "#E0A800",
        "explain": (
            "Chất được chiết xuất từ nhóm 1 hoặc từ thiên nhiên, dùng để "
            "nêm nếm và nấu nướng, ít khi ăn riêng."
        ),
        "examples": "Dầu ăn, bơ, đường, muối, mật ong, giấm.",
    },
    3: {
        "title": "Thực phẩm chế biến",
        "color": "#E8772E",
        "explain": (
            "Kết hợp thực phẩm nhóm 1 với nguyên liệu nhóm 2 (muối, đường, dầu) "
            "để bảo quản hoặc tăng hương vị."
        ),
        "examples": "Bánh mì thủ công, phô mai, cá hộp, rau củ muối, trái cây ngâm.",
    },
    4: {
        "title": "Siêu chế biến",
        "color": "#D63B3B",
        "explain": (
            "Công thức công nghiệp có nhiều thành phần mà bếp nhà ít dùng "
            "như phụ gia, hương liệu, chất tạo màu, nhũ hóa."
        ),
        "examples": "Mì ăn liền, nước ngọt, snack đóng gói, xúc xích công nghiệp.",
    },
}
EXAMPLES = {
    "Mì ăn liền": (
        "wheat flour, palm oil, salt, sugar, flavour enhancers "
        "(monosodium glutamate, disodium inosinate), maltodextrin, "
        "hydrolysed soy protein, colour (caramel), spices"
    ),
    "Bánh mì": "wheat flour, water, yeast, salt, sugar",
    "Nước ngọt": (
        "carbonated water, sugar, colour (caramel E150d), "
        "acidity regulator (phosphoric acid), natural flavourings, caffeine"
    ),
    "Sữa chua": "milk, live cultures",
}
# Các từ khóa này chỉ dùng để minh họa khi phân tích, không trực tiếp quyết định NOVA.
MARKERS = {
    "Mã phụ gia E": r"\be\s?\d{3,4}[a-z]?\b",
    "Hương liệu": r"flavou?r|flavouring|flavoring",
    "Chất tạo màu": r"colou?r|caramel|colouring|coloring",
    "Chất nhũ hóa": r"emulsifier|lecithin|diglycerides|mono.? and diglycerides",
    "Chất làm ngọt": (
        r"sweetener|aspartame|sucralose|acesulfame|glucose[- ]fructose|"
        r"fructose syrup|glucose syrup|saccharin"
    ),
    "Maltodextrin / tinh bột biến tính": r"maltodextrin|modified starch|dextrose",
    "Chất điều vị": r"glutamate|inosinate|guanylate|flavou?r enhancer",
    "Chất bảo quản / điều chỉnh độ acid": (
        r"preservative|acidity regulator|benzoate|sorbate|citric acid"
    ),
}
# ============================================================
# CSS - GIAO DIỆN SÁNG, PHONG CÁCH TẠP CHÍ ẨM THỰC
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root {
  --bg:#FAF7F0; --card:#FFFFFF; --soft:#F1EBDD; --ink:#1D2B24; --muted:#68766E; --line:#E4DDCC;
  --green:#1F5A3B; --green2:#2E7D4F; --tomato:#E4572E; --mustard:#F2B134; --shadow:0 6px 22px rgba(29,43,36,.07);
  --body:var(--body); --flagbg:#FDECE6; --flagline:#F5C4B3; --flagtx:#A63A18; --calbg:#EAF3EC; --calline:#C9E0CF; --hover:#F1EBDD;
}
html, body, .stApp { background:var(--bg) !important; color:var(--ink); font-family:'Plus Jakarta Sans', system-ui, sans-serif; color-scheme:light; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display:none !important; }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1120px; padding-top:1.1rem; padding-bottom:3rem; }
h1, h2, h3, h4 { font-family:Fraunces, Georgia, serif !important; color:var(--ink) !important; font-weight:700 !important; letter-spacing:-.01em; }
p, li, label, [data-testid="stMarkdownContainer"] { color:var(--ink); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color:var(--muted) !important; }
a { color:var(--green2); }
hr { border-color:var(--line) !important; }

/* ---------- Thanh đầu trang & điều hướng ---------- */
.topbar { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; padding:6px 2px 14px; }
.brand { display:flex; align-items:center; gap:11px; }
.brand-mark { display:flex; align-items:center; justify-content:center; width:40px; height:40px; border-radius:12px; font-size:1.25rem; background:var(--green); box-shadow:0 5px 14px rgba(31,90,59,.3); }
.brand-name { font-family:Fraunces, Georgia, serif; font-size:1.3rem; font-weight:800; color:var(--ink); line-height:1; }
.brand-name i { color:var(--tomato); font-style:normal; }
.brand-tag { font-size:.7rem; letter-spacing:1.6px; color:var(--muted); font-weight:700; margin-top:3px; }
.top-chip { padding:6px 12px; border:1px solid var(--line); border-radius:999px; background:var(--card); color:var(--muted); font-size:.76rem; font-weight:600; }
[data-testid="stRadio"] [role="radiogroup"] { gap:8px; flex-wrap:wrap; }
[data-testid="stRadio"] [role="radiogroup"] label {
  padding:8px 16px; border-radius:999px; background:var(--card); border:1px solid var(--line); cursor:pointer; transition:all .15s;
}
[data-testid="stRadio"] [role="radiogroup"] label:hover { border-color:var(--green2); transform:translateY(-1px); }
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) { background:var(--green); border-color:var(--green); }
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) * { color:#fff !important; }
[data-testid="stRadio"] [role="radiogroup"] label > div:first-child { display:none; }
[data-testid="stRadio"] [role="radiogroup"] label p { font-weight:650; font-size:.9rem; }

/* ---------- Hero ---------- */
.hero { background-size:200% 200%; animation:shift 14s ease infinite; display:grid; grid-template-columns:1.25fr 1fr; gap:26px; align-items:center; margin:18px 0 22px; padding:30px 34px; border-radius:26px;
  background:linear-gradient(120deg, #1F5A3B, #2E7D4F); box-shadow:0 18px 40px rgba(31,90,59,.22); position:relative; overflow:hidden; }
.hero::after { content:""; position:absolute; right:-60px; top:-60px; width:220px; height:220px; border-radius:50%; background:rgba(242,177,52,.18); }
.eyebrow { font-size:.72rem; font-weight:800; letter-spacing:2px; color:#F2D98D; }
.hero-title { margin-top:8px; font-family:Fraunces, Georgia, serif; font-size:clamp(1.7rem, 3.4vw, 2.6rem); font-weight:800; line-height:1.12; color:#fff; }
.hero-sub { margin-top:11px; color:#d7eadf; font-size:.97rem; max-width:470px; line-height:1.55; }
.hero-strip { position:relative; z-index:1; display:grid; grid-template-columns:repeat(4, 1fr); gap:9px; }
.hs { padding:14px 8px 12px; border-radius:16px; background:rgba(255,255,255,.96); text-align:center; }
.hs b { display:block; font-family:Fraunces, Georgia, serif; font-size:1.7rem; line-height:1; }
.hs span { display:block; margin-top:6px; font-size:.68rem; font-weight:700; color:var(--muted); line-height:1.25; }
.page-head { margin:14px 0 16px; }
.page-head h2 { margin:0; font-size:1.9rem; }
.page-head p { margin:4px 0 0; color:var(--muted); }
.sec-title { margin:24px 0 10px; font-family:Fraunces, Georgia, serif; font-size:1.2rem; font-weight:700; color:var(--ink); }

/* ---------- Nút & ô nhập ---------- */
.stButton > button, [data-testid="stLinkButton"] a, [data-testid="stDownloadButton"] button {
  border-radius:12px !important; border:1.5px solid var(--green) !important; background:var(--card) !important; color:var(--green) !important;
  font-weight:700 !important; transition:all .15s ease;
}
.stButton > button:hover, [data-testid="stLinkButton"] a:hover, [data-testid="stDownloadButton"] button:hover { background:var(--hover) !important; transform:translateY(-1px); }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
  background:var(--green) !important; color:#fff !important; border-color:var(--green) !important; }
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { filter:brightness(.9); }
.stButton > button:disabled { opacity:.45; }
input, textarea { color:var(--ink) !important; background:var(--card) !important; }
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div { background:var(--card) !important; border-color:var(--line) !important; border-radius:12px !important; }
textarea::placeholder, input::placeholder { color:#9aa79f !important; opacity:1 !important; }
[data-testid="stMetric"] { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:14px 16px; box-shadow:var(--shadow); }
[data-testid="stMetricLabel"] * { color:var(--muted) !important; }
[data-testid="stMetricValue"] { color:var(--green) !important; font-family:Fraunces, Georgia, serif; }
[data-testid="stAlert"] { border-radius:14px; }
[data-testid="stExpander"] details { background:var(--card); border:1px solid var(--line); border-radius:14px; }
[data-testid="stExpander"] summary * { color:var(--ink) !important; font-weight:650; }
[data-testid="stFileUploaderDropzone"] { border-radius:16px; background:var(--card) !important; border:1.5px dashed var(--line) !important; }
[data-testid="stFileUploaderDropzone"] *, [data-testid="stCameraInput"] label * { color:var(--muted) !important; }
[data-testid="stCameraInput"] { border-radius:16px; }
[data-testid="stToggle"] label p { font-weight:650; font-size:.86rem; }
[data-testid="stTable"] table { background:var(--card); color:var(--ink); border-radius:12px; }
[data-testid="stTable"] th { background:var(--soft) !important; color:var(--ink) !important; }
[data-testid="stTable"] td { color:var(--ink) !important; }

/* ---------- Kết quả phân loại ---------- */
.verdict { padding:24px; border-radius:22px; background:var(--card); border:1px solid var(--line); box-shadow:var(--shadow); position:relative; overflow:hidden; }
.verdict::before { content:""; position:absolute; left:0; top:0; bottom:0; width:8px; background:var(--c); }
.verdict-top { display:flex; align-items:center; gap:16px; }
.verdict-num { display:flex; align-items:center; justify-content:center; width:68px; height:68px; flex:0 0 68px; border-radius:20px; background:var(--c); color:#fff;
  font-family:Fraunces, Georgia, serif; font-size:2.4rem; font-weight:800; animation:pop .6s cubic-bezier(.2,1.4,.4,1) both; }
.verdict-kicker { font-size:.7rem; letter-spacing:1.8px; font-weight:800; color:var(--muted); }
.verdict-title { font-family:Fraunces, Georgia, serif; font-size:1.45rem; font-weight:750; color:var(--ink); line-height:1.15; margin-top:2px; }
.verdict-desc { margin-top:14px; color:var(--body); line-height:1.55; }
.scale { display:flex; gap:7px; margin-top:16px; }
.seg { flex:1; text-align:center; padding:9px 0; border-radius:11px; background:var(--soft); color:var(--muted); font-weight:800; transition:all .2s; }
.seg.on { background:var(--c); color:#fff; transform:scale(1.07); box-shadow:0 6px 14px rgba(0,0,0,.16); }
.score-row { margin-top:12px; }
.score-head { display:flex; justify-content:space-between; gap:10px; margin-bottom:5px; color:var(--ink); font-size:.86rem; }
.score-track { height:9px; border-radius:99px; overflow:hidden; background:var(--soft); }
.score-fill { height:100%; border-radius:99px; transform-origin:left center; animation:grow .9s cubic-bezier(.2,.8,.2,1) both; }
.nova-card { height:100%; min-height:150px; padding:16px; border-radius:18px; background:var(--card); border:1px solid var(--line); box-shadow:var(--shadow); }
.nova-number { display:inline-flex; align-items:center; justify-content:center; width:36px; height:36px; border-radius:11px; color:#fff; font-family:Fraunces, Georgia, serif; font-size:1.3rem; font-weight:800; }
.nova-title { margin-top:9px; font-weight:750; color:var(--ink); }
.nova-example { margin-top:6px; color:var(--muted); font-size:.82rem; line-height:1.45; }
.nv { display:flex; gap:15px; padding:18px; margin:10px 0; border:1px solid var(--line); border-radius:18px; background:var(--card); box-shadow:var(--shadow); }
.nv-bar { width:7px; border-radius:99px; flex-shrink:0; }
.nv-title { font-family:Fraunces, Georgia, serif; font-size:1.15rem; font-weight:700; color:var(--ink); }
.nv-text { margin-top:5px; color:var(--body); }
.nv-example { margin-top:6px; color:var(--muted); font-size:.88rem; }
.chip { display:inline-block; padding:6px 11px; margin:3px; border-radius:999px; background:var(--soft); color:var(--ink); font-size:.82rem; font-weight:500; }
.chip.flag { background:var(--flagbg); border:1px solid var(--flagline); color:var(--flagtx); font-weight:650; }
.callout { padding:16px 18px; border-radius:16px; background:var(--calbg); border:1px solid var(--calline); color:var(--ink); line-height:1.6; }
.empty-box { display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:250px; padding:22px; text-align:center; border:2px dashed #D4CBB5; border-radius:22px; background:rgba(255,255,255,.6); color:var(--muted); gap:7px; }
.empty-box .emoji { font-size:2.5rem; } .empty-box b { color:var(--ink); font-size:1.05rem; }
.hist { padding:12px 14px; margin:8px 0 4px; border-radius:14px; background:var(--card); border:1px solid var(--line); font-size:.86rem; color:var(--ink); }
.hist-dot { display:inline-block; width:10px; height:10px; margin-right:7px; border-radius:50%; }
.step { height:100%; padding:18px; border:1px solid var(--line); border-radius:18px; background:var(--card); box-shadow:var(--shadow); }
.step-n { display:flex; align-items:center; justify-content:center; width:32px; height:32px; border-radius:50%; background:var(--green); color:#fff; font-weight:800; }
.step-t { margin-top:10px; font-weight:750; color:var(--ink); } .step-d { margin-top:4px; color:var(--muted); font-size:.86rem; line-height:1.5; }
.pc { display:flex; gap:16px; align-items:center; padding:14px 16px; margin:6px 0 14px; border-radius:18px; border:1px solid var(--line); background:var(--card); box-shadow:var(--shadow); }
.pc-img { width:78px; height:78px; flex:0 0 78px; border-radius:14px; object-fit:cover; background:var(--soft); }
.pc-noimg { display:flex; align-items:center; justify-content:center; width:78px; height:78px; flex:0 0 78px; border-radius:14px; background:var(--soft); font-size:2rem; }
.pc-name { font-family:Fraunces, Georgia, serif; font-size:1.15rem; font-weight:700; color:var(--ink); } .pc-meta { margin-top:3px; color:var(--muted); font-size:.84rem; }
.tips { padding:18px 20px; border-radius:18px; border:1px solid var(--line); background:var(--card); box-shadow:var(--shadow); }
.tips b { color:var(--ink); font-family:Fraunces, Georgia, serif; font-size:1.05rem; }
.tips ul { margin:8px 0 0; padding-left:18px; color:var(--body); font-size:.88rem; line-height:1.65; }
.how { display:flex; gap:12px; align-items:flex-start; margin:0 0 12px; }
.how-n { flex:0 0 28px; height:28px; border-radius:50%; background:var(--mustard); color:#4a3200; font-weight:800; display:flex; align-items:center; justify-content:center; font-size:.85rem; }
.how-t { font-weight:700; color:var(--ink); font-size:.93rem; } .how-d { color:var(--muted); font-size:.82rem; margin-top:1px; line-height:1.45; }
.scan-code { font-variant-numeric:tabular-nums; font-weight:700; color:var(--green); }

/* ---------- Hiệu ứng chờ & màn hình khởi động ---------- */
.nova-loading { display:flex; align-items:center; justify-content:center; gap:14px; width:100%; margin:12px 0; padding:16px 20px; border:1px solid var(--line); border-radius:16px; background:var(--card); box-shadow:var(--shadow); color:var(--ink); font-weight:650; }
.nova-spinner { position:relative; width:38px; height:38px; flex:0 0 38px; }
.nova-spinner-ring, .nova-spinner-ring-inner { position:absolute; box-sizing:border-box; border-radius:50%; animation-timing-function:linear; animation-iteration-count:infinite; }
.nova-spinner-ring { inset:0; border:4px solid var(--line); border-top-color:var(--green); animation-name:spin; animation-duration:1s; }
.nova-spinner-ring-inner { inset:9px; border:3px solid var(--soft); border-top-color:var(--mustard); animation-name:spin-r; animation-duration:.7s; }
.nova-loading-text { color:var(--ink); font-size:.95rem; }
.nova-splash { display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:62vh; gap:18px; text-align:center; animation:fade-in .4s ease both; }
.nova-splash-logo { font-size:3.6rem; line-height:1; animation:float 1.8s ease-in-out infinite; }
.nova-splash-title { font-family:Fraunces, Georgia, serif; font-size:1.7rem; font-weight:800; color:var(--ink); }
.nova-splash-title span { color:var(--green); }
.nova-splash-bar { width:min(340px, 80vw); height:8px; border-radius:99px; overflow:hidden; background:var(--soft); }
.nova-splash-bar-fill { height:100%; border-radius:99px; background:linear-gradient(90deg, var(--green), var(--mustard)); transition:width .35s ease; }
.nova-splash-step { color:var(--muted); font-size:.92rem; min-height:1.3em; }
.nova-dots span { display:inline-block; width:8px; height:8px; margin:0 3px; border-radius:50%; background:var(--green); animation:bounce 1s ease-in-out infinite; }
.nova-dots span:nth-child(2) { animation-delay:.15s; background:var(--mustard); } .nova-dots span:nth-child(3) { animation-delay:.3s; background:var(--tomato); }
.block-container { animation:fade-up .45s ease both; }
@keyframes shift { 0%,100% { background-position:0% 50%; } 50% { background-position:100% 50%; } }
@keyframes spin { to { transform:rotate(360deg); } } @keyframes spin-r { to { transform:rotate(-360deg); } }
@keyframes pop { 0% { opacity:0; transform:scale(.55); } 70% { transform:scale(1.08); } 100% { opacity:1; transform:scale(1); } }
@keyframes grow { from { transform:scaleX(0); } to { transform:scaleX(1); } }
@keyframes fade-in { from { opacity:0; } to { opacity:1; } }
@keyframes fade-up { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
@keyframes float { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-10px); } }
@keyframes bounce { 0%,80%,100% { transform:translateY(0); opacity:.5; } 40% { transform:translateY(-7px); opacity:1; } }
@media (prefers-reduced-motion:reduce) { * { animation:none !important; } }
@media (max-width:820px) { .hero { grid-template-columns:1fr; padding:22px; } .pc { flex-direction:column; align-items:flex-start; } }

/* ---------- Nâng cấp: thống kê, vòng điểm, lưới thẻ sản phẩm ---------- */
.stats { display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin:0 0 20px; }
.stat { padding:14px 16px; border-radius:18px; background:var(--card); border:1px solid var(--line); box-shadow:var(--shadow); transition:transform .18s, box-shadow .18s; }
.stat:hover, .nova-card:hover, .nv:hover, .step:hover, .sc:hover { transform:translateY(-3px); box-shadow:0 14px 30px rgba(0,0,0,.12); }
.stat b { display:block; font-family:Fraunces, Georgia, serif; font-size:1.7rem; line-height:1.1; color:var(--green2); }
.stat span { display:block; margin-top:4px; font-size:.76rem; font-weight:700; letter-spacing:.4px; color:var(--muted); }
.ring { position:relative; display:flex; align-items:center; justify-content:center; width:86px; height:86px; flex:0 0 86px; margin-left:auto; border-radius:50%;
  background:conic-gradient(var(--c) calc(var(--p) * 1%), var(--soft) 0); }
.ring::before { content:""; position:absolute; inset:9px; border-radius:50%; background:var(--card); }
.ring span { position:relative; font-family:Fraunces, Georgia, serif; font-weight:800; font-size:1.3rem; color:var(--ink); line-height:1; text-align:center; }
.ring small { display:block; margin-top:2px; font-family:'Plus Jakarta Sans', sans-serif; font-size:.55rem; font-weight:700; letter-spacing:.8px; color:var(--muted); }
.sc-img { height:150px; margin:-4px -4px 12px; border-radius:14px; background-color:var(--soft); background-size:cover; background-position:center; display:flex; align-items:center; justify-content:center; font-size:2.4rem; }
.sc-name { font-family:Fraunces, Georgia, serif; font-weight:700; font-size:1.02rem; color:var(--ink); line-height:1.25; min-height:2.5em; }
.sc-meta { margin-top:4px; font-size:.8rem; color:var(--muted); }
.sc-ing { margin-top:8px; font-size:.8rem; color:var(--body); line-height:1.45; min-height:3.6em; }
.badge { display:inline-block; margin-top:8px; padding:3px 9px; border-radius:999px; font-size:.7rem; font-weight:800; background:var(--soft); color:var(--muted); }
.badge.warn { background:var(--flagbg); color:var(--flagtx); }
.nv-num { display:inline-flex; align-items:center; justify-content:center; width:44px; height:44px; border-radius:14px; color:#fff; font-family:Fraunces, Georgia, serif; font-size:1.6rem; font-weight:800; }
.foot { margin-top:30px; padding:16px 0 4px; border-top:1px solid var(--line); color:var(--muted); font-size:.8rem; line-height:1.6; }
@media (max-width:820px) { .stats { grid-template-columns:repeat(2, 1fr); } }
</style>
""",
    unsafe_allow_html=True,
)

if st.session_state.get("dark_mode"):
    st.markdown(
        """
<style>
:root { --bg:#0E1914; --card:#16251D; --soft:#1F3228; --ink:#EAF2EC; --muted:#9FB2A7; --line:#2B4235; --green:#2E9B5B; --green2:#5BD08F;
  --shadow:0 6px 22px rgba(0,0,0,.35); --body:#C5D5CB; --flagbg:#3A1F18; --flagline:#6B3325; --flagtx:#FFB8A0; --calbg:#183326; --calline:#2B5A40; --hover:#223A2D; }
.stApp { color-scheme:dark; }
.hs { background:#F4F1E8; }
</style>
""",
        unsafe_allow_html=True,
    )
dark = bool(st.session_state.get("dark_mode"))
ALT = {"bar": "#5BD08F" if dark else "#1F5A3B", "label": "#9FB2A7" if dark else "#68766E",
       "grid": "#2B4235" if dark else "#E4DDCC", "ink": "#EAF2EC" if dark else "#1D2B24", "bg": "#16251D" if dark else "#FFFFFF"}
# ============================================================
# SESSION STATE
# ============================================================
DEFAULT_STATE = {
    "ingredients": "",
    "result": None,
    "history": [],
    "main_page": NAV_ITEMS[0],
    "product_search_submitted": "",
    "product_search_results": [],
    "selected_product": None,
    "barcode_context": "",
    "navigate_to_prediction": False,
    "app_ready": False,
    "scan_nonce": 0,
    "scan_history": [],
    "last_live_scan": "",
    "last_camera_scan": "",
    "dark_mode": False,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value
if st.session_state.get("main_page") not in NAV_ITEMS:
    st.session_state["main_page"] = NAV_ITEMS[0]
# Cờ điều hướng phải được áp dụng TRƯỚC khi tạo radio có key main_page.
if st.session_state.get("navigate_to_prediction"):
    st.session_state["main_page"] = PREDICT_PAGE
    st.session_state["navigate_to_prediction"] = False
# ============================================================
# HIỆU ỨNG CHỜ TÙY CHỈNH
# ============================================================
@contextmanager
def nova_spinner(message: str):
    """Hiển thị hai vòng tròn đồng tâm xoay ngược chiều trong lúc xử lý."""
    loading = st.empty()
    def update(text: str) -> None:
        loading.markdown(
            '<div class="nova-loading">'
            '<span class="nova-spinner" aria-hidden="true">'
            '<span class="nova-spinner-ring"></span>'
            '<span class="nova-spinner-ring-inner"></span>'
            '</span>'
            f'<span class="nova-loading-text">{html.escape(text)}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    update(message)
    try:
        yield update
    finally:
        loading.empty()

# ============================================================
# HÀM TIỆN ÍCH VÀ MÔ HÌNH
# ============================================================
def meta(label: int) -> dict:
    return NOVA_META.get(int(label), NOVA_META[1])
def page_head(title: str, subtitle: str = "") -> None:
    sub_html = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="page-head"><h2>{html.escape(title)}</h2>{sub_html}</div>', unsafe_allow_html=True)
def render_stats() -> None:
    history_items = st.session_state.get("history", [])
    scans = st.session_state.get("scan_history", [])
    total = len(history_items)
    counts = Counter(item["label"] for item in history_items)
    common = f"NOVA {counts.most_common(1)[0][0]}" if total else "—"
    ultra = f"{counts.get(4, 0) * 100 // total}%" if total else "—"
    cards = [(len(scans), "MÃ ĐÃ QUÉT"), (total, "LẦN PHÂN LOẠI"), (common, "NHÓM GẶP NHIỀU"), (ultra, "SIÊU CHẾ BIẾN")]
    body = "".join(f'<div class="stat"><b>{value}</b><span>{label}</span></div>' for value, label in cards)
    st.markdown(f'<div class="stats">{body}</div>', unsafe_allow_html=True)
def flat(value: str) -> str:
    return " ".join(line.strip() for line in value.splitlines() if line.strip())
def split_ingredients(text: str) -> list[str]:
    parts = re.split(r"[,;\n]", text or "")
    return [part.strip(" .*\t\r") for part in parts if part.strip(" .*\t\r")]
def scan_markers(text: str) -> dict:
    found = {}
    lowered = (text or "").lower()
    for name, pattern in MARKERS.items():
        count = len(re.findall(pattern, lowered, flags=re.IGNORECASE))
        if count:
            found[name] = count
    return found
@st.cache_data(show_spinner=False, max_entries=256)
def predict_cached(text: str) -> tuple[int, str, dict]:
    """Cache dự đoán theo nội dung để không chạy lại mô hình với cùng đầu vào."""
    label, description = predict_ingredients(text)
    raw_scores = predict_proba_ingredients(text)
    scores = {int(key): float(value) for key, value in raw_scores.items()}
    return int(label), str(description), scores
def show_startup_splash() -> None:
    """Màn hình chờ lần đầu mở app: nạp mô hình trước để lần dự đoán đầu tiên không bị khựng."""
    placeholder = st.empty()
    steps = [
        (15, "Đang khởi tạo giao diện..."),
        (55, "Đang nạp mô hình TF-IDF + Linear SVC..."),
        (85, "Đang khởi động bộ phân loại NOVA..."),
        (100, "Sẵn sàng!"),
    ]
    def draw(percent: int, text: str) -> None:
        placeholder.markdown(
            '<div class="nova-splash">'
            '<div class="nova-splash-logo">🥗</div>'
            f'<div class="nova-splash-title"><span>{html.escape(APP_NAME)}</span></div>'
            f'<div class="nova-splash-bar"><div class="nova-splash-bar-fill" style="width:{percent}%"></div></div>'
            f'<div class="nova-splash-step">{html.escape(text)}</div>'
            '<div class="nova-dots"><span></span><span></span><span></span></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    draw(*steps[0])
    time.sleep(0.25)
    draw(*steps[1])
    try:
        predict_cached("water, sugar, salt")  # warm-up: nạp mô hình vào bộ nhớ/cache
    except Exception:  # noqa: BLE001
        pass  # nếu lỗi, trang Dự đoán sẽ báo lỗi chi tiết như bình thường
    draw(*steps[2])
    time.sleep(0.25)
    draw(*steps[3])
    time.sleep(0.2)
    placeholder.empty()
    st.session_state["app_ready"] = True
def add_history(label: int, text: str) -> None:
    item = {"label": int(label), "text": str(text)[:500]}
    history = st.session_state.get("history", [])
    if history and history[0].get("text") == item["text"] and history[0].get("label") == item["label"]:
        return
    st.session_state["history"] = [item] + history[:5]
def make_prediction(text: str, product: dict | None = None) -> None:
    cleaned = (text or "").strip()
    if not cleaned:
        st.session_state["result"] = {
            "kind": "warning",
            "message": "Chưa có danh sách thành phần. Bạn hãy đọc nhãn sản phẩm và nhập thành phần vào ô bên dưới.",
        }
        return
    try:
        label, description, scores = predict_cached(cleaned)
        result = {
            "kind": "ok",
            "text": cleaned,
            "label": label,
            "description": description,
            "scores": scores,
        }
        if product:
            result.update({
                "product_name": product.get("name", ""),
                "barcode": product.get("barcode", ""),
                "source_url": product.get("url", ""),
            })
        st.session_state["result"] = result
        add_history(label, cleaned)
    except FileNotFoundError as exc:
        st.session_state["result"] = {
            "kind": "error",
            "message": f"Chưa tìm thấy mô hình đã huấn luyện. Hãy kiểm tra thư mục mô hình. Chi tiết: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        st.session_state["result"] = {
            "kind": "error",
            "message": f"Không thể dự đoán lúc này. Chi tiết: {exc}",
        }
def clear_prediction() -> None:
    st.session_state["ingredients"] = ""
    st.session_state["result"] = None
    st.session_state["selected_product"] = None
    st.session_state["barcode_context"] = ""
def clear_scan_inputs() -> None:
    """Đổi nonce để camera và ô tải ảnh được tạo lại (xóa ảnh đã chọn)."""
    st.session_state["scan_nonce"] = st.session_state.get("scan_nonce", 0) + 1
def clear_manual_barcode() -> None:
    st.session_state["manual_barcode_input"] = ""
def clear_product_search() -> None:
    st.session_state["product_search_name"] = ""
    st.session_state["product_search_submitted"] = ""
    st.session_state["product_search_results"] = []
def use_example(text: str) -> None:
    st.session_state["ingredients"] = text
    st.session_state["result"] = None
    st.session_state["selected_product"] = None
    st.session_state["barcode_context"] = ""
def render_label_panel(result: dict) -> None:
    label = int(result["label"])
    info = meta(label)
    top_pct = int(round(max(0.0, min(1.0, float(result.get("scores", {}).get(label, 0.0)))) * 100))
    segs = "".join(
        f'<div class="seg{" on" if n == label else ""}" style="--c:{NOVA_META[n]["color"]}">{n}</div>' for n in (1, 2, 3, 4)
    )
    st.markdown(
        flat(
            f"""
            <div class="verdict" style="--c:{info['color']}">
              <div class="verdict-top">
                <div class="verdict-num">{label}</div>
                <div><div class="verdict-kicker">NHÓM NOVA DỰ ĐOÁN</div>
                <div class="verdict-title">{html.escape(info['title'])}</div></div>
                <div class="ring" style="--p:{top_pct};--c:{info['color']}"><span>{top_pct}<small>ĐIỂM</small></span></div>
              </div>
              <div class="verdict-desc">{html.escape(info['explain'])}</div>
              <div class="scale">{segs}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sec-title">Điểm số theo từng nhóm</div>', unsafe_allow_html=True)
    st.caption("Điểm được chuẩn hóa từ đầu ra mô hình để so sánh giữa các nhóm; không phải xác suất hay mức độ chắc chắn tuyệt đối.")
    for group, score in sorted(result.get("scores", {}).items()):
        group = int(group)
        score = max(0.0, min(1.0, float(score)))
        st.markdown(
            f"""
            <div class="score-row">
              <div class="score-head"><span>NOVA {group} · {html.escape(meta(group)['title'])}</span><b>{score * 100:.1f}</b></div>
              <div class="score-track"><div class="score-fill" style="width:{score * 100:.1f}%;background:{meta(group)['color']};"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
def render_ladder(active: int | None = None) -> None:
    columns = st.columns(4, gap="small")
    for column, number in zip(columns, (1, 2, 3, 4)):
        info = NOVA_META[number]
        glow = f"border:2px solid {info['color']};box-shadow:0 10px 24px {info['color']}33;" if number == active else ""
        column.markdown(
            f"""
            <div class="nova-card" style="{glow}">
              <div class="nova-number" style="background:{info['color']}">{number}</div>
              <div class="nova-title">{html.escape(info['title'])}</div>
              <div class="nova-example">{html.escape(info['examples'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
def render_analysis(text: str) -> None:
    ingredients = split_ingredients(text)
    markers = scan_markers(text)
    st.markdown('<div class="sec-title">Phân tích thành phần</div>', unsafe_allow_html=True)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Số thành phần", len(ingredients))
    metric_cols[1].metric("Nhóm dấu hiệu", len(markers))
    metric_cols[2].metric("Lần xuất hiện", sum(markers.values()))
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Thành phần đã đọc**")
        if ingredients:
            chips = "".join(f'<span class="chip">{html.escape(item)}</span>' for item in ingredients[:40])
            if len(ingredients) > 40:
                chips += f'<span class="chip">và {len(ingredients) - 40} thành phần khác</span>'
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("Chưa tách được thành phần từ nội dung nhập.")
    with right:
        st.markdown("**Dấu hiệu thường gặp ở thực phẩm siêu chế biến**")
        if markers:
            flags = "".join(f'<span class="chip flag">{html.escape(name)} ×{count}</span>' for name, count in markers.items())
            st.markdown(flags, unsafe_allow_html=True)
        else:
            st.caption("Không tìm thấy các dấu hiệu phụ gia phổ biến trong danh sách đã nhập.")
        st.caption("Đây là bước quét từ khóa để tham khảo, không phải căn cứ duy nhất để xác định NOVA.")
# ============================================================
# TRA CỨU OPEN FOOD FACTS - CÓ CACHE 15 PHÚT
# ============================================================
OFF_FIELDS = (
    "code,product_name,product_name_vi,product_name_en,brands,quantity,"
    "ingredients_text,ingredients_text_vi,ingredients_text_en,"
    "image_front_url,image_url,url,nova_group,categories"
)
def _product_name(product: dict) -> str:
    return (
        product.get("product_name")
        or product.get("product_name_vi")
        or product.get("product_name_en")
        or "Sản phẩm chưa có tên"
    ).strip()
def _product_ingredients(product: dict) -> str:
    return (
        product.get("ingredients_text")
        or product.get("ingredients_text_vi")
        or product.get("ingredients_text_en")
        or ""
    ).strip()
def normalize_product(item: dict) -> dict:
    barcode = str(item.get("code") or "").strip()
    return {
        "barcode": barcode,
        "name": _product_name(item),
        "brand": str(item.get("brands") or "").strip(),
        "quantity": str(item.get("quantity") or "").strip(),
        "ingredients": _product_ingredients(item),
        "image": item.get("image_front_url") or item.get("image_url") or "",
        "url": item.get("url") or (f"https://world.openfoodfacts.org/product/{barcode}" if barcode else "https://world.openfoodfacts.org/"),
        "nova_group": item.get("nova_group"),
        "categories": item.get("categories") or "",
    }
@st.cache_data(ttl=900, show_spinner=False, max_entries=256)
def fetch_product_by_barcode(barcode: str) -> dict | None:
    if requests is None:
        raise RuntimeError("Thiếu requests. Cài bằng: pip install requests")
    url = f"https://world.openfoodfacts.org/api/v2/product/{quote_plus(barcode)}.json"
    response = requests.get(
        url,
        params={"fields": OFF_FIELDS},
        headers={"User-Agent": USER_AGENT},
        timeout=(8, 30),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != 1 or not payload.get("product"):
        return None
    return normalize_product(payload["product"])
SEARCH_MAX_ATTEMPTS = 6          # số lần thử tối đa cho mỗi lượt tìm
SEARCH_TIMEOUT = (8, 45)         # (chờ kết nối, chờ phản hồi) tính bằng giây
SEARCH_RETRY_STATUS = {429, 500, 502, 503, 504}
SEARCH_CACHE_TTL = 900           # giây
SEARCH_CACHE_MAX = 128
@st.cache_resource(show_spinner=False)
def _search_cache() -> dict:
    """Kho cache thủ công. Chỉ lưu kết quả KHÔNG rỗng, và không gọi phần tử Streamlit nào."""
    return {}
def _search_products_once(query: str) -> list[dict]:
    """Một lần gọi Open Food Facts. Hàm thuần, không dùng st.* nên an toàn để thử lại."""
    response = requests.get(
        "https://world.openfoodfacts.org/cgi/search.pl",
        params={
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 24,
            "sort_by": "unique_scans_n",
            "fields": OFF_FIELDS,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=SEARCH_TIMEOUT,
    )
    response.raise_for_status()
    raw_products = response.json().get("products", [])
    query_lower = query.casefold()
    results, seen = [], set()
    for raw in raw_products:
        product = normalize_product(raw)
        barcode = product["barcode"]
        if barcode and barcode in seen:
            continue
        if barcode:
            seen.add(barcode)
        if product["name"] == "Sản phẩm chưa có tên" and not product["brand"]:
            continue
        name_brand = f"{product['name']} {product['brand']}".casefold()
        product["_rank"] = (
            2.0 * SequenceMatcher(None, query_lower, name_brand).ratio()
            + (1.0 if query_lower in name_brand else 0.0)
            + (0.25 if product["ingredients"] else 0.0)
        )
        results.append(product)
    results.sort(key=lambda item: item["_rank"], reverse=True)
    for item in results:
        item.pop("_rank", None)
    return results[:16]
def search_products_online(query: str, on_retry=None) -> list[dict]:
    """Tìm sản phẩm, thử lại nhiều lần cho tới khi có kết quả.
    on_retry(text) chạy ở NGOÀI hàm cache nên được phép cập nhật giao diện."""
    query = (query or "").strip()
    if len(query) < 2:
        return []
    if requests is None:
        raise RuntimeError("Thiếu requests. Cài bằng: pip install requests")
    cache = _search_cache()
    key = query.casefold()
    hit = cache.get(key)
    if hit and time.time() - hit[0] < SEARCH_CACHE_TTL:
        return hit[1]
    last_error = ""
    for attempt in range(SEARCH_MAX_ATTEMPTS):
        if attempt and on_retry:
            on_retry(f"Đang tìm sản phẩm... (lần thử {attempt + 1}/{SEARCH_MAX_ATTEMPTS}, máy chủ đang phản hồi chậm)")
        try:
            results = _search_products_once(query)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in SEARCH_RETRY_STATUS:
                raise
            last_error = f"Máy chủ Open Food Facts đang bận (mã {status})."
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ValueError) as exc:
            last_error = f"Không kết nối được Open Food Facts ({type(exc).__name__}). Hãy kiểm tra Internet."
        else:
            if results:
                if len(cache) >= SEARCH_CACHE_MAX:
                    cache.pop(next(iter(cache)))
                cache[key] = (time.time(), results)
                return results
            # Phản hồi rỗng: đôi khi do máy chủ quá tải, nên thử lại
        if attempt < SEARCH_MAX_ATTEMPTS - 1:
            time.sleep(min(2.0 * (attempt + 1), 8))
    if last_error:
        raise RuntimeError(last_error)
    return []
def web_search_links(query: str) -> list[tuple[str, str]]:
    encoded = quote_plus(query.strip())
    return [
        ("Google", f"https://www.google.com/search?q={encoded}"),
        ("Bing", f"https://www.bing.com/search?q={encoded}"),
        ("DuckDuckGo", f"https://duckduckgo.com/?q={encoded}"),
        ("Google Shopping", f"https://www.google.com/search?tbm=shop&q={encoded}"),
        ("Open Food Facts", f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={encoded}&search_simple=1&action=process"),
    ]
def show_web_search_links(query: str) -> None:
    st.markdown("**Tìm thêm trên Internet**")
    columns = st.columns(5)
    for column, (label, url) in zip(columns, web_search_links(query)):
        with column:
            st.link_button(label, url, use_container_width=True)
def navigate_to_prediction() -> None:
    st.session_state["navigate_to_prediction"] = True
    st.rerun()
def classify_product(product: dict) -> None:
    """Nạp sản phẩm và chuyển đến trang dự đoán; không tự điền dữ liệu giả."""
    st.session_state["selected_product"] = product
    st.session_state["ingredients"] = product.get("ingredients", "")
    st.session_state["barcode_context"] = product.get("barcode", "")
    if product.get("ingredients", "").strip():
        with nova_spinner("Đang phân tích thành phần sản phẩm..."):
            make_prediction(product["ingredients"], product)
    else:
        st.session_state["result"] = {
            "kind": "warning",
            "message": (
                f"Đã tìm thấy {product.get('name', 'sản phẩm')}, nhưng nguồn dữ liệu "
                "chưa có danh sách thành phần. Hãy nhập thành phần đọc từ bao bì để phân loại."
            ),
        }
    navigate_to_prediction()
def predict_from_barcode(barcode: str) -> None:
    """Tra cứu mã vạch. Khi thiếu dữ liệu, chuyển sang nhập thành phần thủ công."""
    digits = re.sub(r"\D", "", str(barcode))
    if len(digits) not in (8, 12, 13, 14):
        st.error("Mã vạch cần có 8, 12, 13 hoặc 14 chữ số (EAN/UPC).")
        return
    if requests is None:
        st.error("Thiếu requests. Cài bằng: pip install requests")
        return
    try:
        with nova_spinner("Đang tra cứu thông tin sản phẩm..."):
            product = fetch_product_by_barcode(digits)
        if product is None:
            st.session_state["selected_product"] = None
            st.session_state["ingredients"] = ""
            st.session_state["barcode_context"] = digits
            st.session_state["result"] = {
                "kind": "warning",
                "message": f"Đã đọc được mã {digits}, nhưng chưa tìm thấy hồ sơ sản phẩm tương ứng trong Open Food Facts. Bạn có thể tìm bằng mã này trên Internet hoặc nhập thành phần từ bao bì.",
            }
            navigate_to_prediction()
            return
        st.session_state["selected_product"] = product
        st.session_state["ingredients"] = product.get("ingredients", "")
        st.session_state["barcode_context"] = digits
        if product.get("ingredients", "").strip():
            with nova_spinner("Đang phân tích thành phần sản phẩm..."):
                make_prediction(product["ingredients"], product)
        else:
            st.session_state["result"] = {
                "kind": "warning",
                "message": f"Đã tìm thấy {product['name']}, nhưng Open Food Facts chưa có danh sách thành phần. Hãy nhập thành phần đọc từ nhãn sản phẩm.",
            }
        navigate_to_prediction()
    except Exception as exc:  # noqa: BLE001
        st.session_state["barcode_context"] = digits
        st.session_state["result"] = {
            "kind": "error",
            "message": f"Không thể kết nối dịch vụ tra cứu lúc này: {exc}",
        }
        st.session_state["navigate_to_prediction"] = True
        st.rerun()
# ============================================================
# GIẢI MÃ MÃ VẠCH - XỬ LÝ THEO TỪNG BƯỚC, DỪNG NGAY KHI THÀNH CÔNG
# ============================================================
def _decode_candidate(image: Image.Image) -> list[str]:
    values = []
    seen = set()
    if zxingcpp is not None:
        try:
            for item in zxingcpp.read_barcodes(np.asarray(image)):
                value = str(item.text).strip()
                if value and value not in seen:
                    seen.add(value)
                    values.append(value)
        except Exception:
            pass
    if not values and zbar_decode is not None:
        try:
            for item in zbar_decode(image):
                value = item.data.decode("utf-8", errors="ignore").strip()
                if value and value not in seen:
                    seen.add(value)
                    values.append(value)
        except Exception:
            pass
    return values
@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def decode_barcode_bytes(image_bytes: bytes) -> list[str]:
    if zxingcpp is None and zbar_decode is None:
        raise RuntimeError("Chưa có bộ giải mã mã vạch. Cài bằng: pip install zxing-cpp")
    with Image.open(io.BytesIO(image_bytes)) as opened:
        base = ImageOps.exif_transpose(opened).convert("RGB")
    # Giới hạn ảnh quá lớn để giảm RAM/thời gian, vẫn giữ đủ chi tiết barcode.
    if max(base.size) > 2400:
        base.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
    # Thử ảnh gốc trước. Chỉ tạo ảnh xử lý nâng cao khi lần đầu thất bại.
    angles = (0, 90, 180, 270)
    for angle in angles:
        rotated = base.rotate(angle, expand=True) if angle else base
        decoded = _decode_candidate(rotated)
        if decoded:
            return decoded
        gray = ImageOps.grayscale(rotated)
        for candidate in (
            ImageOps.autocontrast(gray),
            ImageEnhance.Contrast(gray).enhance(1.8),
        ):
            decoded = _decode_candidate(candidate)
            if decoded:
                return decoded
        # Upscale có kiểm soát cho ảnh nhỏ, thường giúp đọc mã in nhỏ.
        if max(gray.size) < 1100:
            scale = 2 if max(gray.size) >= 550 else 3
            enlarged = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)
            decoded = _decode_candidate(enlarged)
            if decoded:
                return decoded
        # Chỉ dùng ngưỡng nhị phân ở vòng cuối để tránh xử lý dư thừa.
        for threshold in (125, 165):
            binary = gray.point(lambda px, limit=threshold: 255 if px > limit else 0)
            decoded = _decode_candidate(binary)
            if decoded:
                return decoded
    return []
def decode_barcode_image(uploaded_file) -> list[str]:
    return decode_barcode_bytes(uploaded_file.getvalue())
# ============================================================
# XỬ LÝ KẾT QUẢ QUÉT (QR / MÃ VẠCH)
# ============================================================
VALID_BARCODE_LENGTHS = (8, 12, 13, 14)
def classify_scan(text: str) -> tuple[str, str]:
    """Trả về ("barcode", dãy số) nếu lấy được mã sản phẩm, ngược lại ("qr", nội dung)."""
    text = (text or "").strip()
    compact = re.sub(r"\s", "", text)
    if compact.isdigit() and len(compact) in VALID_BARCODE_LENGTHS:
        return "barcode", compact
    # QR chứa đường dẫn/văn bản có kèm mã sản phẩm (vd .../product/8934563...)
    for match in re.findall(r"(?<!\d)(\d{8,14})(?!\d)", text):
        if len(match) in VALID_BARCODE_LENGTHS:
            return "barcode", match
    return "qr", text
def record_scan(value: str, kind: str, source: str) -> None:
    item = {"value": value, "kind": kind, "source": source, "time": time.strftime("%H:%M:%S")}
    history = [h for h in st.session_state.get("scan_history", []) if h["value"] != value]
    st.session_state["scan_history"] = [item] + history[:7]
def process_scanned_text(text: str, source: str) -> None:
    kind, value = classify_scan(text)
    record_scan(value, kind, source)
    if kind == "barcode":
        predict_from_barcode(value)
    else:
        st.info("Đã đọc được mã QR nhưng không chứa mã sản phẩm EAN/UPC.")
        st.code(value, language=None)
        if re.match(r"^https?://", value, flags=re.IGNORECASE):
            st.link_button("Mở liên kết trong mã QR", value)
def handle_live_payload(payload: dict) -> None:
    """Nhận dữ liệu từ camera trực tiếp: mã đã đọc + ảnh chụp tự động."""
    code = str(payload.get("code") or "").strip()
    image_data = str(payload.get("image") or "")
    if image_data:
        try:
            st.image(base64.b64decode(image_data.split(",", 1)[-1]), caption="Ảnh được chụp tự động khi nhận diện mã", use_container_width=True)
        except Exception:  # noqa: BLE001
            pass
    if not code and image_data:
        # Trình duyệt chụp được ảnh nhưng không tự giải mã: giải mã ở phía máy chủ.
        try:
            found = decode_barcode_bytes(base64.b64decode(image_data.split(",", 1)[-1]))
            code = found[0] if found else ""
        except Exception:  # noqa: BLE001
            code = ""
    if not code:
        st.error("Chưa đọc được mã từ ảnh đã chụp. Hãy bấm “Quét lại”.")
        return
    process_scanned_text(code, "Camera trực tiếp")
def reuse_history(index: int) -> None:
    history = st.session_state.get("history", [])
    if 0 <= index < len(history):
        st.session_state["ingredients"] = history[index]["text"]
        st.session_state["result"] = None
        st.session_state["selected_product"] = None
        st.session_state["barcode_context"] = ""
        st.session_state["main_page"] = PREDICT_PAGE
def render_product_card(product: dict) -> None:
    image = product.get("image") or ""
    thumb = (
        f'<img class="pc-img" src="{html.escape(image, quote=True)}" alt="">'
        if image else '<div class="pc-noimg">🥗</div>'
    )
    meta_bits = [
        html.escape(product.get("brand") or "Chưa rõ thương hiệu"),
        html.escape(product.get("quantity") or "Chưa rõ khối lượng"),
        "Mã: " + html.escape(product.get("barcode") or "không có"),
    ]
    st.markdown(
        flat(
            f"""
            <div class="pc">{thumb}
              <div>
                <div class="pc-name">{html.escape(product.get('name', 'Không rõ tên'))}</div>
                <div class="pc-meta">{' · '.join(meta_bits)}</div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
def result_report(result: dict) -> str:
    info = meta(result["label"])
    lines = [
        "KẾT QUẢ PHÂN LOẠI NOVA",
        f"Sản phẩm: {result.get('product_name') or '(nhập thủ công)'}",
        f"Mã vạch: {result.get('barcode') or '-'}",
        f"Nhóm dự đoán: NOVA {result['label']} - {info['title']}",
        "",
        "Điểm tương đối từng nhóm:",
    ]
    lines += [f"  NOVA {g}: {s * 100:.1f}" for g, s in sorted(result.get("scores", {}).items())]
    lines += ["", "Thành phần:", result.get("text", ""), "", "Chỉ mang tính tham khảo, không thay thế tư vấn dinh dưỡng."]
    return "\n".join(lines)
# ============================================================
# MÀN HÌNH KHỞI ĐỘNG (chỉ chạy một lần cho mỗi phiên)
# ============================================================
if not st.session_state.get("app_ready"):
    show_startup_splash()
# ============================================================
# BỘ QUÉT CAMERA TRỰC TIẾP (nhúng trong file này, không cần file ngoài)
# ============================================================
SCANNER_HTML = r'''<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --teal:#2dd4bf; --mint:#5eead4; --panel:#10232e; --text:#f1f5f9; --muted:#a9bdc9; --ok:#22c55e; --err:#f87171; }
* { box-sizing:border-box; }
html, body { margin:0; padding:0; background:transparent; font-family:Inter, system-ui, -apple-system, "Segoe UI", sans-serif; color:var(--text); }
.wrap { border:1px solid rgba(31,90,59,.4); border-radius:20px; overflow:hidden; background:#07131c; }
.stage { position:relative; width:100%; aspect-ratio:4/3; max-height:460px; background:#000; overflow:hidden; }
video, #snap { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; background:#000; }
#snap { display:none; }
.idle { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px; color:var(--muted); text-align:center; padding:20px; font-size:.92rem; background:radial-gradient(circle at 50% 40%, #12303b, #07131c); }
.idle .big { font-size:2.6rem; }
.frame { position:absolute; left:10%; right:10%; top:24%; bottom:24%; border-radius:16px; pointer-events:none; display:none;
  border:2px solid rgba(94,234,212,.85); box-shadow:0 0 0 999px rgba(0,0,0,.40), 0 0 22px rgba(45,212,191,.35) inset; }
.frame::after { content:""; position:absolute; left:6%; right:6%; top:8%; height:2px; border-radius:2px;
  background:linear-gradient(90deg, transparent, #5eead4, transparent); box-shadow:0 0 12px #2dd4bf;
  animation:sweep 1.7s ease-in-out infinite alternate; }
@keyframes sweep { to { top:90%; } }
.stage.scanning .frame { display:block; }
.stage.found .frame { border-color:var(--ok); box-shadow:0 0 0 999px rgba(0,0,0,.45), 0 0 26px rgba(34,197,94,.55) inset; }
.stage.found .frame::after { display:none; }
.flash { position:absolute; inset:0; background:#fff; opacity:0; pointer-events:none; }
.flash.go { animation:flash .45s ease-out; }
@keyframes flash { 0% { opacity:.85; } 100% { opacity:0; } }
.badge { position:absolute; left:50%; bottom:12px; transform:translateX(-50%); display:none; padding:7px 14px; border-radius:999px; font-weight:700; font-size:.85rem; background:rgba(34,197,94,.92); color:#04210f; white-space:nowrap; }
.stage.found .badge { display:block; }
.bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:10px 12px; border-top:1px solid rgba(148,163,184,.14); background:#0b1b25; }
.status { flex:1 1 180px; display:flex; align-items:center; gap:8px; font-size:.86rem; color:var(--muted); min-width:0; }
.dot { width:9px; height:9px; border-radius:50%; background:#64748b; flex:0 0 9px; }
.dot.live { background:var(--teal); box-shadow:0 0 8px var(--teal); animation:pulse 1.2s infinite; }
.dot.ok { background:var(--ok); } .dot.err { background:var(--err); }
@keyframes pulse { 50% { opacity:.35; } }
button { font:inherit; font-size:.84rem; font-weight:700; color:#fff; background:#0f766e; border:1px solid #2dd4bf; border-radius:10px; padding:8px 12px; cursor:pointer; transition:background .15s, transform .15s; }
button:hover { background:#115e59; transform:translateY(-1px); }
button.ghost { background:rgba(15,118,110,.18); }
button[hidden] { display:none; }
.result { display:none; padding:12px 14px; border-top:1px solid rgba(34,197,94,.3); background:rgba(34,197,94,.08); }
.result.show { display:block; }
.result .lbl { font-size:.74rem; letter-spacing:1px; color:#86efac; font-weight:800; }
.result .val { margin-top:3px; font-size:1.15rem; font-weight:800; word-break:break-all; font-variant-numeric:tabular-nums; }
.result .fmt { margin-top:2px; font-size:.8rem; color:var(--muted); }
.zoom { display:none; align-items:center; gap:8px; padding:8px 12px; border-top:1px solid rgba(148,163,184,.14); background:#0b1b25; font-size:.82rem; color:var(--muted); }
.zoom.show { display:flex; }
.zoom input { flex:1; accent-color:#2dd4bf; }
.hint { display:none; padding:9px 12px; font-size:.82rem; color:#fde68a; background:rgba(234,179,8,.1); border-top:1px solid rgba(234,179,8,.25); }
.hint.show { display:block; }
@media (prefers-reduced-motion: reduce) { .frame::after, .dot.live { animation:none; } }
</style>
</head>
<body>
<div class="wrap">
  <div class="stage" id="stage">
    <video id="video" playsinline muted></video>
    <img id="snap" alt="Ảnh đã chụp">
    <div class="idle" id="idle"><div class="big">📷</div><div id="idleText">Nhấn “Bật camera” để bắt đầu quét</div></div>
    <div class="frame"></div>
    <div class="flash" id="flash"></div>
    <div class="badge" id="badge">✓ Đã tự động chụp</div>
  </div>
  <div class="bar">
    <div class="status"><span class="dot" id="dot"></span><span id="statusText">Chưa bật camera</span></div>
    <button id="btnStart">▶ Bật camera</button>
    <button id="btnShot" class="ghost" hidden>📸 Chụp ngay</button>
    <button id="btnTorch" class="ghost" hidden>🔦 Đèn</button>
    <button id="btnFlip" class="ghost" hidden>🔄 Đổi camera</button>
    <button id="btnAgain" class="ghost" hidden>↺ Quét lại</button>
  </div>
  <div class="zoom" id="zoomRow"><span>🔍 Zoom</span><input type="range" id="zoom" min="1" max="1" step="0.1" value="1"><span id="zoomVal">1×</span></div>
  <div class="hint" id="hint">Chưa đọc được? Đưa máy xa/gần hơn, tăng sáng hoặc zoom. Bạn cũng có thể bấm “📸 Chụp ngay” để máy chủ giải mã ảnh.</div>
  <div class="result" id="result">
    <div class="lbl">ĐÃ NHẬN DIỆN</div>
    <div class="val" id="resVal"></div>
    <div class="fmt" id="resFmt"></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const video = $("video"), snap = $("snap"), stage = $("stage");
const work = document.createElement("canvas");
const wctx = work.getContext("2d", { willReadFrequently: true });

let stream = null, track = null, running = false, facing = "environment";
let detector = null, zx = null, timer = null, torchOn = false;
let lastVal = "", streak = 0, lastNonce = null, started = false;

function post(type, data) {
  window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type }, data || {}), "*");
}
function fit() { post("streamlit:setFrameHeight", { height: document.documentElement.scrollHeight + 6 }); }
function status(text, state) {
  $("statusText").textContent = text;
  $("dot").className = "dot" + (state ? " " + state : "");
  fit();
}

/* ---------- Kiểm tra checksum EAN/UPC để loại đọc nhầm ---------- */
function gtinOk(s) {
  if (!/^(\d{8}|\d{12,14})$/.test(s)) return false;
  const d = s.split("").map(Number), chk = d.pop();
  let sum = 0;
  d.reverse().forEach((n, i) => { sum += n * (i % 2 === 0 ? 3 : 1); });
  return (10 - (sum % 10)) % 10 === chk;
}
function valid(f) {
  const fmt = f.format.toLowerCase();
  return ["ean_13", "ean_8", "upc_a"].includes(fmt) ? gtinOk(f.text) : true;
}

/* ---------- Bộ giải mã: BarcodeDetector (nhanh) + ZXing cục bộ (luôn có) ---------- */
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = src; el.onload = resolve; el.onerror = () => reject(new Error("Không tải được " + src));
    document.head.appendChild(el);
  });
}
async function initDecoder() {
  if (detector || zx) return;
  if ("BarcodeDetector" in window) {
    try {
      const want = ["ean_13", "ean_8", "upc_a", "upc_e", "code_128", "code_39", "itf", "qr_code"];
      const sup = await BarcodeDetector.getSupportedFormats();
      const use = want.filter(f => sup.includes(f));
      if (use.length) detector = new BarcodeDetector({ formats: use });
    } catch (e) { detector = null; }
  }
  // ZXing nạp từ CDN (2 nguồn). Nếu trình duyệt có BarcodeDetector thì vẫn quét được kể cả khi CDN lỗi.
  const CDNS = ["https://cdn.jsdelivr.net/npm/@zxing/library@0.21.3/umd/index.min.js", "https://unpkg.com/@zxing/library@0.21.3/umd/index.min.js"];
  if (!window.ZXing) { for (const u of CDNS) { try { await loadScript(u); break; } catch (e) {} } }
  if (!window.ZXing) {
    if (detector) return;
    throw new Error("Không tải được bộ giải mã. Hãy dùng “📸 Chụp ngay” hoặc chế độ “Tải ảnh”.");
  }
  const Z = window.ZXing, hints = new Map();
  hints.set(Z.DecodeHintType.POSSIBLE_FORMATS, [
    Z.BarcodeFormat.EAN_13, Z.BarcodeFormat.EAN_8, Z.BarcodeFormat.UPC_A, Z.BarcodeFormat.UPC_E,
    Z.BarcodeFormat.CODE_128, Z.BarcodeFormat.CODE_39, Z.BarcodeFormat.ITF, Z.BarcodeFormat.QR_CODE,
  ]);
  hints.set(Z.DecodeHintType.TRY_HARDER, true);
  zx = new Z.MultiFormatReader();
  zx.setHints(hints);
}
/* crop = 1 là toàn khung hình; <1 là cắt vùng giữa (mã nhỏ sẽ lớn hơn, dễ đọc hơn) */
function zxDecode(crop) {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) return null;
  const cw = vw * crop, ch = vh * (crop === 1 ? 1 : crop * 0.85);
  const sx = (vw - cw) / 2, sy = (vh - ch) / 2;
  const k = Math.min(1, 1280 / Math.max(cw, ch));
  work.width = Math.round(cw * k); work.height = Math.round(ch * k);
  wctx.drawImage(video, sx, sy, cw, ch, 0, 0, work.width, work.height);
  const Z = window.ZXing;
  try {
    const bmp = new Z.BinaryBitmap(new Z.HybridBinarizer(new Z.HTMLCanvasElementLuminanceSource(work)));
    const r = zx.decodeWithState(bmp);
    return { text: r.getText(), format: String(Z.BarcodeFormat[r.getBarcodeFormat()] || "unknown") };
  } catch (e) { return null; }
}

/* ---------- Vòng lặp nhận diện: luân phiên các cách đọc để tăng tỷ lệ thành công ---------- */
let frameNo = 0, scanStart = 0;
async function tick() {
  if (!running) return;
  frameNo++;
  let found = null;
  try {
    if (detector) {
      const r = await detector.detect(video);
      if (r.length) found = { text: r[0].rawValue, format: r[0].format };
    }
    if (!found && zx) {
      // Luân phiên: toàn khung → cắt giữa 70% → cắt giữa 50%
      found = zxDecode([1, 0.7, 0.5][frameNo % 3]);
    }
  } catch (e) { /* bỏ qua khung hình lỗi */ }

  if (found && found.text) {
    if (found.text === lastVal) streak++; else { lastVal = found.text; streak = 1; }
    const need = found.format.toLowerCase() === "qr_code" ? 1 : 2;
    if (streak >= need && valid(found)) return capture(found, "auto");
  } else if (frameNo % 4 === 0) {
    streak = 0;
  }
  if (Date.now() - scanStart > 10000) $("hint").classList.add("show");
  timer = setTimeout(tick, 90);
}

/* ---------- Tự động chụp khi đọc được mã ---------- */
function beep() {
  try {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const o = ac.createOscillator(), g = ac.createGain();
    o.type = "sine"; o.frequency.value = 880; g.gain.value = 0.08;
    o.connect(g); g.connect(ac.destination); o.start();
    setTimeout(() => { o.stop(); ac.close(); }, 130);
  } catch (e) {}
}
function grabFrame() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const k = Math.min(1, 1280 / Math.max(vw, vh));
  const c = document.createElement("canvas");
  c.width = Math.round(vw * k); c.height = Math.round(vh * k);
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.85);
}
function capture(f, source) {
  const img = grabFrame();
  snap.src = img; snap.style.display = "block";
  stage.classList.remove("scanning"); stage.classList.add("found");
  const fl = $("flash"); fl.classList.remove("go"); void fl.offsetWidth; fl.classList.add("go");
  beep(); if (navigator.vibrate) navigator.vibrate(80);
  stopCamera();

  const code = f ? f.text : "";
  const fmt = f ? f.format.toUpperCase() : "ẢNH CHỤP TAY";
  $("badge").textContent = source === "auto" ? "✓ Đã tự động chụp" : "✓ Đã chụp";
  $("resVal").textContent = code || "Đang giải mã trên máy chủ…";
  $("resFmt").textContent = "Loại: " + fmt + " · đã lưu ảnh chụp";
  $("result").classList.add("show");
  $("hint").classList.remove("show");
  $("btnStart").hidden = true; $("btnShot").hidden = true; $("btnAgain").hidden = false;
  status(code ? "Đã nhận diện, đang xử lý…" : "Đã chụp, đang giải mã…", "ok");
  post("streamlit:setComponentValue", {
    value: { code, format: fmt, image: img, source, ts: Date.now() },
    dataType: "json",
  });
}

/* ---------- Camera ---------- */
async function startCamera() {
  if (running) return;
  resetView();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    status("Trình duyệt không cho dùng camera (cần HTTPS hoặc localhost).", "err"); return;
  }
  status("Đang mở camera…");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
  } catch (e) {
    status("Không mở được camera. Hãy cấp quyền camera cho trình duyệt.", "err");
    $("idleText").textContent = "Camera bị chặn hoặc không có thiết bị. Dùng chế độ “Chụp ảnh” hoặc “Tải ảnh”.";
    return;
  }
  video.srcObject = stream;
  try { await video.play(); } catch (e) {}
  track = stream.getVideoTracks()[0];
  const caps = track.getCapabilities ? track.getCapabilities() : {};
  // Lấy nét liên tục nếu thiết bị hỗ trợ (giúp đọc mã vạch nhanh hơn rõ rệt)
  try {
    if (caps.focusMode && caps.focusMode.includes("continuous"))
      await track.applyConstraints({ advanced: [{ focusMode: "continuous" }] });
  } catch (e) {}
  // Thanh zoom
  if (caps.zoom && caps.zoom.max > caps.zoom.min) {
    const z = $("zoom");
    z.min = caps.zoom.min; z.max = caps.zoom.max; z.step = caps.zoom.step || 0.1; z.value = caps.zoom.min;
    $("zoomVal").textContent = (+z.value).toFixed(1) + "×";
    $("zoomRow").classList.add("show");
  } else { $("zoomRow").classList.remove("show"); }

  $("idle").style.display = "none";
  stage.classList.add("scanning");
  $("btnTorch").hidden = !caps.torch;
  $("btnFlip").hidden = false; $("btnShot").hidden = false; $("btnStart").hidden = true;
  running = true; lastVal = ""; streak = 0; frameNo = 0; scanStart = Date.now();
  status("Đưa mã QR / mã vạch vào khung, app sẽ tự chụp…", "live");
  try { await initDecoder(); } catch (e) { status(e.message, "err"); stopCamera(); return; }
  tick();
}
function stopCamera() {
  running = false; clearTimeout(timer);
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; track = null; torchOn = false;
  video.srcObject = null;
  $("btnTorch").hidden = true; $("btnFlip").hidden = true; $("btnShot").hidden = true;
  $("zoomRow").classList.remove("show");
}
function resetView() {
  snap.style.display = "none"; snap.removeAttribute("src");
  stage.classList.remove("found");
  $("result").classList.remove("show"); $("hint").classList.remove("show");
  $("btnAgain").hidden = true;
}

$("btnStart").onclick = startCamera;
$("btnShot").onclick = () => { if (running) capture(null, "manual"); };
$("zoom").oninput = async e => {
  $("zoomVal").textContent = (+e.target.value).toFixed(1) + "×";
  if (track) { try { await track.applyConstraints({ advanced: [{ zoom: +e.target.value }] }); } catch (err) {} }
};
$("btnAgain").onclick = () => { $("idle").style.display = "flex"; startCamera(); };
$("btnFlip").onclick = async () => { facing = facing === "environment" ? "user" : "environment"; stopCamera(); await startCamera(); };
$("btnTorch").onclick = async () => {
  if (!track) return;
  torchOn = !torchOn;
  try { await track.applyConstraints({ advanced: [{ torch: torchOn }] }); } catch (e) { torchOn = false; }
};
window.addEventListener("pagehide", stopCamera);

/* ---------- Giao thức component của Streamlit ---------- */
window.addEventListener("message", ev => {
  if (!ev.data || ev.data.type !== "streamlit:render") return;
  const args = ev.data.args || {};
  if (lastNonce !== null && args.nonce !== lastNonce) { stopCamera(); resetView(); $("idle").style.display = "flex"; started = false; }
  lastNonce = args.nonce;
  if (!started) { started = true; if (args.autostart) startCamera(); else status("Chưa bật camera"); }
  fit();
});
post("streamlit:componentReady", { apiVersion: 1 });
fit();
</script>
</body>
</html>
'''

@st.cache_resource(show_spinner=False)
def _load_scanner():
    """Ghi giao diện quét ra thư mục tạm rồi khai báo component, nên app chỉ cần đúng một file."""
    digest = hashlib.md5(SCANNER_HTML.encode("utf-8")).hexdigest()[:10]
    folder = Path(tempfile.gettempdir()) / f"nova_scanner_{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text(SCANNER_HTML, encoding="utf-8")
    return components.declare_component(f"nova_barcode_scanner_{digest}", path=str(folder))
live_scanner = _load_scanner()
# ============================================================
# ĐẦU TRANG VÀ ĐIỀU HƯỚNG
# ============================================================
head_l, head_r = st.columns([4, 1.25])
with head_l:
    st.markdown(
        """
        <div class="topbar">
          <div class="brand"><div class="brand-mark">🥗</div>
            <div><div class="brand-name">Nova<i>Check</i></div><div class="brand-tag">PHÂN LOẠI THỰC PHẨM THEO NOVA</div></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with head_r:
    st.toggle("🌙 Chế độ tối", key="dark_mode")
selected_page = st.radio("Điều hướng", NAV_ITEMS, key="main_page", horizontal=True, label_visibility="collapsed")
# ============================================================
# TRANG 1: QUÉT MÃ VẠCH
# ============================================================
if selected_page == SCAN_PAGE:
    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="eyebrow">QUÉT · TRA CỨU · PHÂN LOẠI</div>
            <div class="hero-title">Quét một mã,<br>biết ngay mức độ chế biến.</div>
            <div class="hero-sub">Đưa mã QR hoặc mã vạch lên camera. Ứng dụng tự chụp, tra cứu thành phần và cho bạn biết sản phẩm thuộc nhóm NOVA nào.</div>
          </div>
          <div class="hero-strip">
            <div class="hs"><b style="color:#2E9B5B">1</b><span>Tươi,<br>ít chế biến</span></div>
            <div class="hs"><b style="color:#E0A800">2</b><span>Nguyên liệu<br>nấu ăn</span></div>
            <div class="hs"><b style="color:#E8772E">3</b><span>Chế biến</span></div>
            <div class="hs"><b style="color:#D63B3B">4</b><span>Siêu<br>chế biến</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_stats()
    scan_col, help_col = st.columns([1.45, 1], gap="large")
    nonce = st.session_state.get("scan_nonce", 0)
    pending_lookup = None
    with scan_col:
        modes = ["Quét trực tiếp", "Chụp ảnh", "Tải ảnh", "Nhập mã"]
        mode = st.radio("Chế độ quét", modes, key="scan_mode", horizontal=True, label_visibility="collapsed")
        # ---- Chế độ 1: camera trực tiếp, tự chụp khi thấy mã ----
        if mode == modes[0]:
            payload = live_scanner(nonce=nonce, autostart=True, key=f"live_scanner_{nonce}", default=None)
            if payload and str(payload.get("ts")) != st.session_state.get("last_live_scan"):
                st.session_state["last_live_scan"] = str(payload.get("ts"))
                handle_live_payload(payload)
            st.caption("Cần HTTPS hoặc localhost để trình duyệt cho phép camera. Nếu bị chặn, hãy chuyển sang “Chụp ảnh” hoặc “Tải ảnh”.")
        # ---- Chế độ 2: chụp bằng camera của trình duyệt (tự giải mã sau khi chụp) ----
        elif mode == modes[1]:
            camera_image = st.camera_input("Chụp mã bằng camera", key=f"barcode_camera_{nonce}")
            if camera_image is not None:
                fingerprint = hashlib.sha256(camera_image.getvalue()).hexdigest()
                if st.session_state.get("last_camera_scan") != fingerprint:
                    st.session_state["last_camera_scan"] = fingerprint
                    try:
                        with nova_spinner("Đang tự động nhận diện mã từ ảnh vừa chụp..."):
                            codes = decode_barcode_image(camera_image)
                        if not codes:
                            st.error("Chưa đọc được mã. Hãy chụp lại gần hơn, đủ sáng, giữ mã thẳng và tránh lóa.")
                        else:
                            st.success("Đã nhận diện: " + ", ".join(codes))
                            process_scanned_text(codes[0], "Chụp ảnh")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Không xử lý được ảnh camera: {exc}")
                st.button("↺ Chụp lại", key="clear_camera_image", on_click=clear_scan_inputs, use_container_width=True)
        # ---- Chế độ 3: tải ảnh từ máy ----
        elif mode == modes[2]:
            uploaded_image = st.file_uploader(
                "Tải ảnh có mã QR / mã vạch (PNG, JPG, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"barcode_image_upload_{nonce}",
            )
            if uploaded_image is not None:
                st.image(uploaded_image, caption="Ảnh đã tải lên", use_container_width=True)
                up_cols = st.columns([3, 1])
                with up_cols[0]:
                    decode_clicked = st.button("Đọc mã từ ảnh", key="decode_uploaded_barcode_button", type="primary", use_container_width=True)
                with up_cols[1]:
                    st.button("Xóa ảnh", key="clear_scan_image", on_click=clear_scan_inputs, use_container_width=True)
                if decode_clicked:
                    try:
                        with nova_spinner("Đang nhận diện mã..."):
                            codes = decode_barcode_image(uploaded_image)
                        if not codes:
                            st.error("Chưa đọc được mã. Hãy chọn ảnh rõ hơn hoặc nhập mã thủ công.")
                        else:
                            st.success("Đã nhận diện: " + ", ".join(codes))
                            process_scanned_text(codes[0], "Tải ảnh")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Không xử lý được ảnh: {exc}")
        # ---- Chế độ 4: nhập tay ----
        else:
            manual_barcode = st.text_input(
                "Dãy số dưới mã vạch (EAN/UPC)",
                key="manual_barcode_input",
                placeholder="Ví dụ: 893xxxxxxxxxx",
            )
            manual_cols = st.columns([2, 1])
            with manual_cols[0]:
                lookup_clicked = st.button("Tra cứu mã đã nhập", key="lookup_manual_barcode", type="primary",
                                           use_container_width=True, disabled=not manual_barcode.strip())
            with manual_cols[1]:
                st.button("Xóa", key="clear_manual_barcode_btn", on_click=clear_manual_barcode,
                          use_container_width=True, disabled=not manual_barcode.strip())
            if lookup_clicked:
                digits = re.sub(r"\D", "", manual_barcode)
                if len(digits) not in VALID_BARCODE_LENGTHS:
                    st.warning("Mã EAN/UPC thường có 8, 12, 13 hoặc 14 chữ số.")
                else:
                    record_scan(digits, "barcode", "Nhập tay")
                    predict_from_barcode(digits)
    with help_col:
        st.markdown(
            """
            <div class="tips"><b>Cách hoạt động</b>
              <div style="height:10px"></div>
              <div class="how"><div class="how-n">1</div><div><div class="how-t">Quét mã</div><div class="how-d">Camera tự chụp khi đọc được mã QR hoặc mã vạch.</div></div></div>
              <div class="how"><div class="how-n">2</div><div><div class="how-t">Tra cứu sản phẩm</div><div class="how-d">Lấy tên và thành phần từ Open Food Facts.</div></div></div>
              <div class="how"><div class="how-n">3</div><div><div class="how-t">Phân loại NOVA</div><div class="how-d">Mô hình đọc thành phần và cho biết nhóm 1–4.</div></div></div>
            </div>
            <div style="height:12px"></div>
            <div class="tips"><b>Mẹo quét nhanh</b>
              <ul>
                <li>Đưa mã vào giữa khung, giữ máy chắc tay 1–2 giây.</li>
                <li>Đủ sáng, tránh lóa; dùng 🔦 nếu máy có đèn.</li>
                <li>Mã khó đọc: dùng zoom hoặc nút “📸 Chụp ngay”.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        scans = st.session_state.get("scan_history", [])
        if scans:
            st.markdown('<div class="sec-title">Quét gần đây</div>', unsafe_allow_html=True)
            for i, scan in enumerate(scans[:4]):
                row = st.columns([3, 1.4], gap="small")
                label = scan["value"] if len(scan["value"]) <= 22 else scan["value"][:20] + "…"
                row[0].markdown(
                    f'<span class="scan-code">{html.escape(label)}</span><br>'
                    f'<span style="color:var(--muted);font-size:.78rem">{html.escape(scan["source"])} · {scan["time"]}</span>',
                    unsafe_allow_html=True,
                )
                if scan["kind"] == "barcode" and row[1].button("Tra lại", key=f"rescan_{i}", use_container_width=True):
                    pending_lookup = scan["value"]
    if pending_lookup:
        predict_from_barcode(pending_lookup)
    if st.session_state.get("barcode_context"):
        st.divider()
        st.caption(f"Mã vạch gần nhất: {st.session_state['barcode_context']}")
        show_web_search_links(st.session_state["barcode_context"] + " thành phần sản phẩm")
# ============================================================
# TRANG 2: TÌM SẢN PHẨM
# ============================================================
elif selected_page == SEARCH_PAGE:
    page_head("Tìm sản phẩm", "Tìm theo tên hoặc thương hiệu trong Open Food Facts, rồi chọn sản phẩm để phân loại.")
    with st.form("product_search_form", clear_on_submit=False):
        query_col, button_col, clear_col = st.columns([5, 1.2, 1])
        with query_col:
            product_query = st.text_input(
                "Tên sản phẩm hoặc thương hiệu",
                key="product_search_name",
                placeholder="Ví dụ: mì Hảo Hảo, Coca-Cola, sữa chua Vinamilk...",
            )
        with button_col:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Tìm kiếm", type="primary", use_container_width=True)
        with clear_col:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.form_submit_button(" Xóa", on_click=clear_product_search, use_container_width=True)
    if submitted:
        st.session_state["product_search_submitted"] = product_query.strip()
        st.session_state["product_search_results"] = []
    query = st.session_state.get("product_search_submitted", "")
    if query:
        show_web_search_links(query + " thực phẩm thành phần")
        st.markdown('<div class="sec-title">Kết quả từ Open Food Facts</div>', unsafe_allow_html=True)
        try:
            with nova_spinner("Đang tìm sản phẩm...") as set_message:
                products = search_products_online(query, on_retry=set_message)
            st.session_state["product_search_results"] = products
        except Exception as exc:  # noqa: BLE001
            products = []
            st.warning(f"Chưa thể lấy dữ liệu Open Food Facts: {exc}")
        if not products:
            st.info("Chưa tìm thấy sản phẩm phù hợp trong cơ sở dữ liệu. Bạn có thể mở các trang tìm kiếm bên trên hoặc nhập thành phần thủ công ở mục Dự đoán NOVA.")
        else:
            st.caption(f"Tìm thấy {len(products)} kết quả phù hợp. Vui lòng kiểm tra đúng tên, thương hiệu và khối lượng trước khi phân loại.")
            for start in range(0, len(products), 3):
                for offset, (column, product) in enumerate(zip(st.columns(3, gap="medium"), products[start:start + 3])):
                    index = start + offset
                    with column, st.container(border=True):
                        image = product.get("image") or ""
                        bg = f"background-image:url('{html.escape(image, quote=True)}');" if image else ""
                        placeholder = "" if image else "🥗"
                        preview = product["ingredients"][:140] + ("…" if len(product["ingredients"]) > 140 else "") if product.get("ingredients") else ""
                        badge = (
                            '<span class="badge">Có thành phần</span>' if preview
                            else '<span class="badge warn">Chưa có thành phần</span>'
                        )
                        nova_src = product.get("nova_group")
                        if nova_src in (1, 2, 3, 4, "1", "2", "3", "4"):
                            badge += f' <span class="badge">NOVA nguồn: {nova_src}</span>'
                        st.markdown(
                            flat(
                                f"""
                                <div class="sc">
                                  <div class="sc-img" style="{bg}">{placeholder}</div>
                                  <div class="sc-name">{html.escape(product['name'])}</div>
                                  <div class="sc-meta">{html.escape(product['brand'] or 'Chưa rõ thương hiệu')} · {html.escape(product['quantity'] or 'Chưa rõ khối lượng')}</div>
                                  <div class="sc-ing">{html.escape(preview)}</div>
                                  {badge}
                                </div>
                                """
                            ),
                            unsafe_allow_html=True,
                        )
                        actions = st.columns(2)
                        with actions[0]:
                            if st.button("Phân loại", key=f"choose_product_{index}_{product['barcode']}", type="primary", use_container_width=True):
                                classify_product(product)
                        with actions[1]:
                            st.link_button("Nguồn", product["url"], use_container_width=True)
# ============================================================
# TRANG 3: DỰ ĐOÁN NOVA
# ============================================================
elif selected_page == PREDICT_PAGE:
    page_head("Phân loại thành phần", "Dán thành phần trên nhãn, hoặc chỉnh sửa nội dung lấy từ mã vạch / kết quả tìm kiếm.")
    selected_product = st.session_state.get("selected_product")
    if selected_product:
        render_product_card(selected_product)
        if selected_product.get("url"):
            st.markdown(f"[Mở nguồn dữ liệu sản phẩm]({selected_product['url']})")
        if not selected_product.get("ingredients", "").strip():
            search_terms = f"{selected_product.get('name', '')} {selected_product.get('brand', '')} {selected_product.get('barcode', '')} thành phần ingredients"
            show_web_search_links(search_terms)
    elif st.session_state.get("barcode_context"):
        barcode_value = st.session_state["barcode_context"]
        st.info(f"Mã vạch đã nhận diện: {barcode_value}. Nguồn Open Food Facts chưa có dữ liệu đầy đủ; bạn có thể tra cứu thêm hoặc nhập thành phần từ bao bì.")
        show_web_search_links(barcode_value + " ingredients thành phần sản phẩm")
    input_col, result_col = st.columns([1.05, 0.95], gap="large")
    with input_col:
        st.text_area(
            "Thành phần trên nhãn",
            key="ingredients",
            height=230,
            placeholder="Ví dụ: sugar, wheat flour, vegetable oil, salt...",
            help="Giữ nguyên thứ tự trên nhãn và nhập càng đầy đủ càng tốt.",
        )
        action_cols = st.columns([2, 1])
        with action_cols[0]:
            predict_clicked = st.button("Dự đoán", type="primary", use_container_width=True)
        with action_cols[1]:
            st.button("Xóa nội dung", key="clear_prediction", on_click=clear_prediction, use_container_width=True)
        st.markdown('<div class="sec-title">Thử nhanh với ví dụ</div>', unsafe_allow_html=True)
        example_cols = st.columns(2)
        for column, (name, example_text) in zip(example_cols, EXAMPLES.items()):
            with column:
                st.button(name, key=f"example_{name}", on_click=use_example, args=(example_text,), use_container_width=True)
        with st.expander("Mẹo nhập thành phần để kết quả tốt hơn"):
            st.markdown(
                "- Nhập đủ danh sách thành phần, không chỉ chép vài dòng đầu.\n"
                "- Giữ nguyên thứ tự xuất hiện trên bao bì.\n"
                "- Mô hình được huấn luyện trên văn bản thành phần; với nhãn tiếng Việt, kết quả có thể kém ổn định hơn văn bản tiếng Anh.\n"
                "- Kết quả là dự đoán hỗ trợ tham khảo, không thay thế đánh giá chuyên môn về quy trình sản xuất."
            )
    with result_col:
        if predict_clicked:
            current_text = st.session_state.get("ingredients", "")
            product_for_result = selected_product
            if selected_product and current_text.strip() != selected_product.get("ingredients", "").strip():
                product_for_result = None
            with nova_spinner("Đang chuẩn hóa thành phần và dự đoán NOVA..."):
                make_prediction(current_text, product_for_result)
        result = st.session_state.get("result")
        if result is None:
            st.markdown(
                '<div class="empty-box"><span class="emoji">🥗</span><b>Chưa có kết quả</b>'
                'Nhập thành phần ở bên trái rồi bấm Dự đoán.</div>',
                unsafe_allow_html=True,
            )
        elif result.get("kind") == "warning":
            st.warning(result.get("message", "Chưa có dữ liệu để dự đoán."))
        elif result.get("kind") == "error":
            st.error(result.get("message", "Đã xảy ra lỗi khi dự đoán."))
        elif result.get("kind") == "ok":
            if result.get("product_name"):
                st.markdown(f"**Sản phẩm:** {html.escape(result['product_name'])}")
            render_label_panel(result)
            st.download_button(
                "⬇️ Tải kết quả (.txt)",
                data=result_report(result),
                file_name="nova_result.txt",
                mime="text/plain",
                use_container_width=True,
            )
    current_result = st.session_state.get("result")
    if current_result and current_result.get("kind") == "ok":
        render_analysis(current_result["text"])
        st.markdown('<div class="sec-title">Bốn nhóm NOVA</div>', unsafe_allow_html=True)
        render_ladder(current_result["label"])
# ============================================================
# TRANG: LỊCH SỬ
# ============================================================
elif selected_page == HISTORY_PAGE:
    page_head("Lịch sử phiên này", "Các lần phân loại và quét mã gần đây. Dữ liệu chỉ lưu trong phiên hiện tại.")
    render_stats()
    h_left, h_right = st.columns(2, gap="large")
    with h_left:
        st.markdown('<div class="sec-title">Phân loại gần đây</div>', unsafe_allow_html=True)
        past = st.session_state.get("history", [])
        if not past:
            st.caption("Chưa có lần phân loại nào.")
        for idx, item in enumerate(past):
            info = meta(item["label"])
            st.markdown(
                f'<div class="hist"><span class="hist-dot" style="background:{info["color"]}"></span>'
                f'<b>NOVA {item["label"]} · {html.escape(info["title"])}</b><br>'
                f'<span style="color:var(--muted)">{html.escape(item["text"][:110])}</span></div>',
                unsafe_allow_html=True,
            )
            st.button("Dùng lại thành phần này", key=f"reuse_{idx}", on_click=reuse_history, args=(idx,), use_container_width=True)
        if past and st.button("Xóa lịch sử phân loại", key="clear_history", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()
    history_lookup = None
    with h_right:
        st.markdown('<div class="sec-title">Mã đã quét</div>', unsafe_allow_html=True)
        scans = st.session_state.get("scan_history", [])
        if not scans:
            st.caption("Chưa quét mã nào.")
        for i, scan in enumerate(scans):
            row = st.columns([3, 1.4], gap="small")
            row[0].markdown(
                f'<div class="hist"><span class="scan-code">{html.escape(scan["value"][:40])}</span><br>'
                f'<span style="color:var(--muted);font-size:.78rem">{html.escape(scan["source"])} · {scan["time"]}</span></div>',
                unsafe_allow_html=True,
            )
            if scan["kind"] == "barcode" and row[1].button("Tra lại", key=f"hist_rescan_{i}", use_container_width=True):
                history_lookup = scan["value"]
        if scans and st.button("Xóa lịch sử quét", key="clear_scan_history_page", use_container_width=True):
            st.session_state["scan_history"] = []
            st.rerun()
    if history_lookup:
        predict_from_barcode(history_lookup)
# ============================================================
# TRANG 4: HIỂU VỀ NOVA
# ============================================================
elif selected_page == LEARN_PAGE:
    page_head("Kiến thức NOVA", "NOVA phân loại thực phẩm theo mức độ và mục đích chế biến, không dựa vào lượng dinh dưỡng.")
    numbers = sorted(NOVA_META)
    for pair in (numbers[0:2], numbers[2:4]):
        for column, number in zip(st.columns(2, gap="medium"), pair):
            m = NOVA_META[number]
            column.markdown(
                flat(
                    f"""
                    <div class="nv" style="flex-direction:column;gap:10px;height:100%">
                      <div class="nv-num" style="background:{m['color']}">{number}</div>
                      <div class="nv-title">{m['title']}</div>
                      <div class="nv-text">{m['explain']}</div>
                      <div class="nv-example">Ví dụ: {m['examples']}</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
    st.markdown('<div class="sec-title">Mẹo đọc nhãn</div>', unsafe_allow_html=True)
    with st.expander("Thành phần được sắp xếp thế nào?"):
        st.write("Thành phần được liệt kê theo khối lượng giảm dần, nên những thứ đứng đầu danh sách chiếm nhiều nhất.")
    with st.expander("Những dấu hiệu nào hay gặp ở nhóm 4?"):
        st.write(
            "Danh sách dài, có mã phụ gia (E…), hương liệu, chất tạo màu, chất nhũ hóa, "
            "chất điều vị, maltodextrin hoặc siro glucose-fructose, tức là những thứ bếp nhà ít khi dùng."
        )
    with st.expander("NOVA có thay thế bảng thành phần dinh dưỡng không?"):
        st.write(
            "Không. NOVA chỉ nói về mức độ chế biến. Để đánh giá một sản phẩm, "
            "bạn vẫn nên xem thêm lượng đường, muối, chất béo và năng lượng."
        )
    st.caption(
        "Phân loại thực tế cần xét cả công thức, thành phần và quy trình chế biến, "
        "không chỉ một nguyên liệu riêng lẻ."
    )
# ============================================================
# TRANG 5: THÔNG TIN MÔ HÌNH
# ============================================================
elif selected_page == MODEL_PAGE:
    page_head("Mô hình", "Cách mô hình hoạt động và kết quả thực nghiệm.")
    st.markdown("#### Mô hình hoạt động như thế nào")
    steps = [
        ("Chuẩn hóa", "Đưa về chữ thường, loại bỏ dấu."),
        ("TF-IDF", "Biến danh sách thành phần thành vector, dùng unigram và bigram."),
        ("Linear SVC", "Tính điểm cho từng nhóm NOVA."),
        ("Kết quả", "Nhóm có điểm cao nhất được chọn làm dự đoán."),
    ]
    for col, (i, (t, d)) in zip(st.columns(4, gap="medium"), enumerate(steps, start=1)):
        col.markdown(
            f'<div class="step"><div class="step-n">{i}</div>'
            f'<div class="step-t">{t}</div><div class="step-d">{d}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="sec-title">Kết quả thực nghiệm</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "95,76%")
    c2.metric("Macro-F1", "87,96%")
    c3.metric("Weighted-F1", "95,79%")
    chart_col, text_col = st.columns([3, 2], gap="large")
    with chart_col:
        chart_data = pd.DataFrame({
            "Chỉ số": ["Accuracy", "Macro-F1", "Weighted-F1"],
            "Điểm": [95.76, 87.96, 95.79]
        })
        chart = alt.Chart(chart_data).mark_bar(
            color=ALT["bar"],
            cornerRadiusEnd=4
            ).encode(
            x=alt.X(
                "Điểm:Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(
                    title=None,
                    labelColor=ALT["label"],
                    gridColor=ALT["grid"],
                    tickColor=ALT["grid"]
                )
            ),
            y=alt.Y(
                "Chỉ số:N",
                sort=None,
                axis=alt.Axis(
                    title=None,
                    labelColor=ALT["ink"],
                    domain=False,
                    ticks=False
                )
            ),
            tooltip=["Chỉ số", "Điểm"]
        ).properties(
            height=180,
            background=ALT["bg"]
        ).configure_view(
            stroke="transparent"
        ).configure_axis(
            labelFont="Inter",
            labelFontSize=12
        )
        st.altair_chart(chart, use_container_width=True)
    with text_col:
        st.markdown(
            flat(
                """
                <div class="callout">
                📊 Mô hình Linear SVC đạt Macro-F1 87,96%.
                NOVA 2 là nhóm có F1-score thấp nhất do số lượng
                mẫu ít hơn đáng kể so với các nhóm còn lại.
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    st.markdown("#### Cấu hình thực nghiệm")
    config = pd.DataFrame(
        {
            "Mục": [
                "Tổng số mẫu",
                "Tập huấn luyện",
                "Tập kiểm tra",
                "Random state",
                "Tỷ lệ kiểm tra",
                "Mô hình tốt nhất",
                "Đặc trưng"
            ],
            "Giá trị": [
                "988.924",
                "791.139",
                "197.785",
                "42",
                "20%",
                "Linear SVC",
                "TF-IDF văn bản thành phần"
            ],
        }
    ).set_index("Mục")
    st.table(config)
    st.markdown("#### Đánh giá chi tiết từng nhóm NOVA")
    report = pd.DataFrame({
        "NOVA": ["NOVA 1", "NOVA 2", "NOVA 3", "NOVA 4"],
        "Precision": [0.8714, 0.7102, 0.9057, 0.9866],
        "Recall": [0.9288, 0.7487, 0.9113, 0.9767],
        "F1-score": [0.8992, 0.7289, 0.9085, 0.9816],
    })
    st.table(report)
    st.caption(
        "Kết quả được đánh giá trên tập kiểm tra gồm 197.785 mẫu "
        "sau khi chia dữ liệu theo tỷ lệ 80/20."
    )
    st.markdown("#### Hạn chế cần biết")
    st.markdown(
        "- Mô hình chỉ đọc văn bản nên chưa hiểu quy trình chế biến thực tế.\n"
        "- Nhãn trong Open Food Facts do cộng đồng đóng góp, có thể có sai lệch."
    )
# ============================================================
# CHÂN TRANG
# ============================================================
st.markdown(
    '<div class="foot">Kết quả chỉ mang tính tham khảo cho mục đích học tập, không thay thế tư vấn dinh dưỡng. '
    'Dữ liệu sản phẩm lấy từ Open Food Facts.<br>Python · Streamlit · TF-IDF · Linear SVC</div>',
    unsafe_allow_html=True,
)
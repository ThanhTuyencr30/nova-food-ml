import html
import io
import re
import time
from difflib import SequenceMatcher
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
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
    initial_sidebar_state="expanded",
)

APP_NAME = "NOVA Food Classifier"
USER_AGENT = "NOVA-Food-Classifier/2.0 (educational project)"
NAV_ITEMS = [
    "📷 Quét mã vạch",
    "🔎 Tìm sản phẩm",
    "🔮 Dự đoán NOVA",
    "📚 Hiểu về NOVA",
    "🧠 Về mô hình",
]


# ============================================================
# DỮ LIỆU NOVA
# ============================================================
NOVA_META = {
    1: {
        "title": "Chưa hoặc ít chế biến",
        "color": "#22c55e",
        "explain": (
            "Thực phẩm tự nhiên hoặc chỉ qua xử lý tối thiểu như rửa, sấy, "
            "xay, đông lạnh, không thêm chất phụ gia."
        ),
        "examples": "Rau củ, trái cây, thịt cá tươi, trứng, sữa tươi, gạo, đậu.",
    },
    2: {
        "title": "Nguyên liệu chế biến",
        "color": "#45beea",
        "explain": (
            "Chất được chiết xuất từ nhóm 1 hoặc từ thiên nhiên, dùng để "
            "nêm nếm và nấu nướng, ít khi ăn riêng."
        ),
        "examples": "Dầu ăn, bơ, đường, muối, mật ong, giấm.",
    },
    3: {
        "title": "Thực phẩm chế biến",
        "color": "#f97316",
        "explain": (
            "Kết hợp thực phẩm nhóm 1 với nguyên liệu nhóm 2 (muối, đường, dầu) "
            "để bảo quản hoặc tăng hương vị."
        ),
        "examples": "Bánh mì thủ công, phô mai, cá hộp, rau củ muối, trái cây ngâm.",
    },
    4: {
        "title": "Siêu chế biến",
        "color": "#ef4444",
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
# CSS - MỘT BỘ GIAO DIỆN DUY NHẤT, TRÁNH CSS BỊ GHI ĐÈ LẪN NHAU
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --bg: #07131c;
  --panel: #10232e;
  --panel2: #15313b;
  --line: rgba(148, 163, 184, .16);
  --teal: #2dd4bf;
  --mint: #5eead4;
  --text: #f1f5f9;
  --muted: #a9bdc9;
}
html, body, .stApp {
  background: radial-gradient(circle at 85% 0%, rgba(45,212,191,.09), transparent 27%),
              linear-gradient(180deg, #07131c, #0b1720 75%);
  color: var(--text);
  font-family: Inter, sans-serif;
}
.block-container { max-width: 1380px; padding-top: 1.3rem; padding-bottom: 2rem; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #10232e; border-right: 1px solid rgba(94,234,212,.12); }
[data-testid="stSidebar"] * { color: #e7f0f5; }
[data-testid="stCaptionContainer"] { color: #a9bdc9 !important; }
p, li, label { color: #e2eaf0; }
h1, h2, h3, h4 { color: #67e8d4 !important; font-weight: 750 !important; }
.hero {
  position: relative; overflow: hidden; padding: 28px 32px; margin-bottom: 20px;
  border: 1px solid rgba(45,212,191,.2); border-radius: 22px;
  background: linear-gradient(120deg, rgba(15,118,110,.25), rgba(21,49,59,.86));
  box-shadow: 0 14px 35px rgba(0,0,0,.18);
}
.hero-tag { color: #5eead4; font-size: .78rem; font-weight: 800; letter-spacing: 1.3px; }
.hero-title { margin-top: 7px; font-size: clamp(1.55rem, 3vw, 2.35rem); font-weight: 800; line-height: 1.2; color: #f8fafc; }
.hero-sub { margin-top: 9px; color: #c6d8e0; max-width: 780px; }
.pill { display:inline-block; margin: 12px 6px 0 0; padding: 6px 11px; border:1px solid rgba(94,234,212,.2); border-radius: 999px; background:rgba(7,19,28,.35); color:#c9f7ef; font-size:.8rem; }
.sec-title { margin: 22px 0 10px; font-size: 1.15rem; font-weight: 750; color: #67e8d4; }
.stButton > button, [data-testid="stLinkButton"] a {
  border-radius: 11px !important; border: 1px solid #2dd4bf !important;
  background: #0f766e !important; color: #fff !important; font-weight: 700 !important;
  transition: background .15s ease, transform .15s ease;
}
.stButton > button:hover, [data-testid="stLinkButton"] a:hover { background:#115e59 !important; transform:translateY(-1px); }
input, textarea, [data-baseweb="select"] > div {
  color:#f8fafc !important; background:#18313d !important; border-color:rgba(45,212,191,.35) !important;
}
textarea::placeholder, input::placeholder { color:#9fb3c1 !important; opacity:1 !important; }
[data-testid="stMetric"] { background:#10232e; border:1px solid rgba(94,234,212,.13); border-radius:14px; padding:12px 14px; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:#f1f5f9 !important; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stExpander"] details { background:rgba(16,35,46,.65); border:1px solid rgba(94,234,212,.14); border-radius:13px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color:#e7f0f5 !important; }
.result-card { padding:22px; border-radius:19px; border:1px solid rgba(94,234,212,.2); background:linear-gradient(145deg,#15313b,#10232e); }
.result-badge { display:inline-block; border-radius:999px; padding:10px 18px; font-size:1.55rem; font-weight:850; }
.result-title { margin-top:15px; font-size:1.25rem; font-weight:750; color:#f8fafc; }
.result-desc { margin-top:7px; color:#cad8df; }
.score-row { margin-top:13px; }
.score-head { display:flex; justify-content:space-between; gap:10px; margin-bottom:5px; color:#e6eef2; font-size:.88rem; }
.score-track { height:10px; border-radius:99px; overflow:hidden; background:#263d49; }
.score-fill { height:100%; border-radius:99px; }
.nova-card { height:100%; min-height:150px; padding:17px; border-radius:15px; background:#10232e; border:1px solid rgba(148,163,184,.13); border-top:4px solid; }
.nova-number { font-size:1.8rem; font-weight:850; }
.nova-title { margin-top:5px; font-weight:750; }
.nova-example { margin-top:8px; color:#a9bdc9; font-size:.83rem; }
.nv { display:flex; gap:15px; padding:17px; margin:10px 0; border:1px solid rgba(148,163,184,.13); border-radius:15px; background:rgba(16,35,46,.65); }
.nv-bar { width:6px; border-radius:99px; flex-shrink:0; }
.nv-title { font-size:1.03rem; font-weight:750; color:#f1f5f9; }
.nv-text { margin-top:5px; color:#cad8df; }
.nv-example { margin-top:5px; color:#a9bdc9; font-size:.88rem; }
.chip { display:inline-block; padding:6px 10px; margin:3px; border-radius:999px; background:#1b3945; color:#e7f6f5; font-size:.82rem; }
.chip.flag { background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.25); color:#fecaca; }
.empty-box { display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:250px; padding:22px; text-align:center; border:1px dashed rgba(94,234,212,.35); border-radius:18px; background:rgba(16,35,46,.45); color:#b9cbd4; gap:7px; }
.empty-box .emoji { font-size:2.5rem; }
.empty-box b { color:#f1f5f9; font-size:1.05rem; }
.hist { padding:8px 0; border-bottom:1px solid rgba(148,163,184,.12); font-size:.83rem; }
.hist-dot { display:inline-block; width:9px; height:9px; margin-right:7px; border-radius:50%; }
.step { height:100%; padding:16px; border:1px solid rgba(94,234,212,.15); border-radius:15px; background:#10232e; }
.step-n { display:flex; align-items:center; justify-content:center; width:31px; height:31px; border-radius:50%; background:#2dd4bf; color:#06211f; font-weight:850; }
.step-t { margin-top:10px; font-weight:750; }
.step-d { margin-top:4px; color:#a9bdc9; font-size:.85rem; }
@media(max-width:700px){ .hero {padding:20px;} .hero-title{font-size:1.6rem;} }
</style>
""",
    unsafe_allow_html=True,
)

MIN_LOADING_SECONDS = 2.2


def render_loader():
    return """
    <style>
    .nova-loader {
        padding: 30px 20px;
        margin: 15px 0;
        border-radius: 20px;
        text-align: center;
        background: linear-gradient(145deg, #10232e, #15313b);
        border: 1px solid rgba(94,234,212,.25);
    }

    .ld-ring {
        width: 88px;
        height: 88px;
        border: 5px solid rgba(45,212,191,.15);
        border-top: 5px solid #2dd4bf;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto;
        position: relative;
    }

    .ld-ring-inner {
        position: absolute;
        inset: 12px;
        border: 4px solid transparent;
        border-bottom-color: #5eead4;
        border-radius: 50%;
        animation: spin 1s linear infinite reverse;
    }

    .ld-icon {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .ld-title {
        color: #5eead4;
        font-size: 20px;
        font-weight: 800;
        margin-top: 20px;
    }

    .ld-message {
        color: #cbd5e1;
        margin-top: 12px;
        animation: fade 1.2s infinite alternate;
    }

    @keyframes fade {
        from { opacity: .45; }
        to { opacity: 1; }
    }

    .ld-progress {
        height: 8px;
        margin-top: 22px;
        border-radius: 20px;
        background: #263d49;
        overflow: hidden;
    }

    .ld-fill {
        width: 0%;
        height: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg,#0d9488,#2dd4bf,#a7f3d0);
        animation: fill 2.2s linear forwards;
    }

    @keyframes fill {
        to { width: 100%; }
    }

    .ld-rabbit {
        font-size: 27px;
        display: inline-block;
        animation: rabbit 1s ease-in-out infinite alternate;
    }

    @keyframes rabbit {
        from { transform: translateX(-65px); }
        to { transform: translateX(65px); }
    }
    </style>

    <div class="nova-loader">
        <div style="position:relative;width:88px;margin:auto">
            <div class="ld-ring">
                <div class="ld-ring-inner"></div>
            </div>
            <div class="ld-icon">🥗</div>
        </div>

        <div class="ld-title">NOVA FOOD CLASSIFIER</div>

        <div class="ld-message">
            Đang chuẩn bị phân tích thành phần...
        </div>

        <div class="ld-progress">
            <div class="ld-fill"></div>
        </div>

        <div style="margin-top:15px">
            <span class="ld-rabbit">🐇</span>
        </div>

        <div style="color:#a9bdc9;font-size:13px">
            Vui lòng chờ trong giây lát!
        </div>
    </div>
    """


def run_prediction_with_loader(text, product=None):
    loader_slot = st.empty()
    start_time = time.time()

    if st.session_state.get("fx", True):
        loader_slot.markdown(
            render_loader(),
            unsafe_allow_html=True
        )

    make_prediction(text, product)

    elapsed = time.time() - start_time
    remaining = MIN_LOADING_SECONDS - elapsed

    if st.session_state.get("fx", True) and remaining > 0:
        time.sleep(remaining)

    loader_slot.empty()
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
    "scan_feedback": None,
    "pending_prediction": False,
    "fx": True,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value
if st.session_state.get("main_page") not in NAV_ITEMS:
    st.session_state["main_page"] = NAV_ITEMS[0]

# Cờ điều hướng phải được áp dụng TRƯỚC khi tạo radio có key main_page.
if st.session_state.get("navigate_to_prediction"):
    st.session_state["main_page"] = "🔮 Dự đoán NOVA"
    st.session_state["navigate_to_prediction"] = False


# ============================================================
# HÀM TIỆN ÍCH VÀ MÔ HÌNH
# ============================================================
def meta(label: int) -> dict:
    return NOVA_META.get(int(label), NOVA_META[1])


def flat(value: str) -> str:
    return " ".join(line.strip() for line in value.splitlines() if line.strip())


def split_ingredients(text: str) -> list[str]:
    parts = re.split(r"[,;\n]", text or "")
    return [part.strip(" .\t\r") for part in parts if part.strip(" .\t\r")]


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


def use_example(text: str) -> None:
    st.session_state["ingredients"] = text
    st.session_state["result"] = None
    st.session_state["selected_product"] = None
    st.session_state["barcode_context"] = ""


def render_label_panel(result: dict) -> None:
    label = int(result["label"])
    info = meta(label)
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-badge" style="background:{info['color']}22;color:{info['color']};">🔥 NOVA {label}</div>
          <div class="result-title">{html.escape(info['title'])}</div>
          <div class="result-desc">{html.escape(info['explain'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Điểm số tương đối theo từng nhóm NOVA**")
    st.caption("Điểm được chuẩn hóa từ đầu ra mô hình để so sánh giữa các nhóm; không phải xác suất hay mức độ chắc chắn tuyệt đối.")
    for group, score in sorted(result.get("scores", {}).items()):
        group = int(group)
        score = max(0.0, min(1.0, float(score)))
        st.markdown(
            f"""
            <div class="score-row">
              <div class="score-head"><span>NOVA {group}</span><b>{score * 100:.1f} điểm</b></div>
              <div class="score-track"><div class="score-fill" style="width:{score \* 100:.1f}%;background:{meta(group)['color']};"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ladder(active: int | None = None) -> None:
    columns = st.columns(4, gap="small")
    for column, number in zip(columns, (1, 2, 3, 4)):
        info = NOVA_META[number]
        glow = f"box-shadow:0 0 18px {info['color']}44;" if number == active else ""
        column.markdown(
            f"""
            <div class="nova-card" style="border-top-color:{info['color']};{glow}">
              <div class="nova-number" style="color:{info['color']}">{number}</div>
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
        timeout=(5, 12),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != 1 or not payload.get("product"):
        return None
    return normalize_product(payload["product"])


@st.cache_data(ttl=900, show_spinner=False, max_entries=128)
def search_products_online(query: str) -> list[dict]:
    if requests is None:
        raise RuntimeError("Thiếu requests. Cài bằng: pip install requests")
    query = query.strip()
    if len(query) < 2:
        return []
    endpoint = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_terms": query,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 24,
        "sort_by": "unique_scans_n",
        "fields": OFF_FIELDS,
    }
    response = requests.get(endpoint, params=params, headers={"User-Agent": USER_AGENT}, timeout=(5, 15))
    response.raise_for_status()
    raw_products = response.json().get("products", [])
    results = []
    seen = set()
    query_lower = query.casefold()

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
    st.session_state["selected_product"] = product
    st.session_state["ingredients"] = product.get("ingredients", "")
    st.session_state["barcode_context"] = product.get("barcode", "")

    if product.get("ingredients", "").strip():
        st.session_state["result"] = None
        st.session_state["pending_prediction"] = True
    else:
        st.session_state["pending_prediction"] = False
        st.session_state["result"] = {
            "kind": "warning",
            "message": (
                f"Đã tìm thấy {product.get('name', 'sản phẩm')}, "
                "nhưng chưa có danh sách thành phần. "
                "Hãy nhập thành phần đọc từ bao bì để phân loại."
            ),
        }

    navigate_to_prediction()
    
def predict_from_barcode(barcode: str) -> None:
    digits = re.sub(r"\D", "", str(barcode))

    if len(digits) not in (8, 12, 13, 14):
        st.session_state["scan_feedback"] = {
            "type": "error",
            "message": "Mã vạch cần có 8, 12, 13 hoặc 14 chữ số."
        }
        return

    if requests is None:
        st.session_state["scan_feedback"] = {
            "type": "error",
            "message": "Thiếu requests. Hãy cài đặt bằng pip install requests."
        }
        return

    try:
        product = fetch_product_by_barcode(digits)

        if product is None:
            st.session_state["selected_product"] = None
            st.session_state["ingredients"] = ""
            st.session_state["barcode_context"] = digits
            st.session_state["pending_prediction"] = False
            st.session_state["result"] = None

            st.session_state["scan_feedback"] = {
                "type": "warning",
                "message": (
                    f"Đã đọc được mã vạch {digits}, nhưng chưa tìm thấy "
                    "sản phẩm trong Open Food Facts. "
                    "Bạn có thể tìm kiếm bằng mã này hoặc nhập thành phần "
                    "từ bao bì để tiếp tục."
                )
            }
            return

        st.session_state["selected_product"] = product
        st.session_state["ingredients"] = product.get("ingredients", "")
        st.session_state["barcode_context"] = digits
        st.session_state["scan_feedback"] = None

        if product.get("ingredients", "").strip():
            st.session_state["result"] = None
            st.session_state["pending_prediction"] = True
        else:
            st.session_state["pending_prediction"] = False
            st.session_state["result"] = {
                "kind": "warning",
                "message": (
                    f"Đã tìm thấy {product['name']}, "
                    "nhưng chưa có danh sách thành phần. "
                    "Hãy nhập thành phần từ nhãn sản phẩm."
                )
            }

        navigate_to_prediction()

    except Exception as exc:
        st.session_state["scan_feedback"] = {
            "type": "error",
            "message": f"Không thể kết nối dịch vụ tra cứu: {exc}"
        }

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
# SIDEBAR - ĐIỀU HƯỚNG VÀ LỊCH SỬ
# ============================================================
with st.sidebar:
    st.markdown("## 🥗 NOVA CLASSIFIER")
    st.caption("Hiểu thực phẩm qua danh sách thành phần")
    st.divider()
    st.markdown("**DANH MỤC CHỨC NĂNG**")
    selected_page = st.radio(
        "Điều hướng",
        NAV_ITEMS,
        key="main_page",
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**TỐI ƯU**")
    st.caption("Tra cứu và giải mã được lưu cache tạm thời để hạn chế gọi lại dịch vụ khi chuyển trang.")
    st.divider()
    st.markdown("**LỊCH SỬ DỰ ĐOÁN**")
    history = st.session_state.get("history", [])
    if not history:
        st.caption("Chưa có dự đoán nào trong phiên này.")
    else:
        for item in history[:6]:
            info = meta(item["label"])
            snippet = html.escape(item["text"][:48])
            st.markdown(
                f'<div class="hist"><span class="hist-dot" style="background:{info["color"]}"></span>'
                f'<b>NOVA {item["label"]}</b><br>{snippet}</div>',
                unsafe_allow_html=True,
            )
        if st.button("Xóa lịch sử", key="clear_history", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()
    st.divider()
    st.caption("Python · Streamlit · TF-IDF · Linear SVC")
    st.caption("Đồ án học thuật · NOVA Food Processing Classification")

# ============================================================
# HEADER CHUNG
# ============================================================
st.markdown(
    """
    <div class="hero">
      <div class="hero-tag">🥗 NOVA.CLASSIFIER</div>
      <div class="hero-title">Ăn uống thông minh bắt đầu từ việc hiểu thực phẩm!</div>
      <div class="hero-sub">Tra cứu sản phẩm, đọc danh sách thành phần và phân loại theo bốn nhóm NOVA.</div>
      <span class="pill">Quét mã vạch</span>
      <span class="pill">Tìm sản phẩm</span>
      <span class="pill">TF-IDF</span>
      <span class="pill">Linear SVC</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TRANG 1: QUÉT MÃ VẠCH
# ============================================================
if selected_page == "📷 Quét mã vạch":
    feedback = st.session_state.get("scan_feedback")

if feedback:
    if feedback["type"] == "error":
        st.error(feedback["message"], icon="🚫")
    else:
        st.warning(feedback["message"], icon="⚠️")
    st.markdown("## Quét mã vạch sản phẩm")
    st.caption(
        "Nhập mã sản phẩm hoặc sử dụng camera, tải ảnh để nhận diện mã vạch."
    )

    # HIỂN THỊ THÔNG BÁO NGAY TRÊN TRANG QUÉT
    feedback = st.session_state.get("scan_feedback")

    if feedback:
        if feedback["type"] == "error":
            st.error(feedback["message"])
        else:
            st.warning(feedback["message"])

    st.markdown("### ⌨️ 1. Nhập mã vạch bằng bàn phím")

    manual_col, button_col = st.columns([4, 1])

    with manual_col:
        manual_barcode = st.text_input(
            "Mã EAN/UPC",
            key="manual_barcode_input",
            placeholder="Ví dụ: 893xxxxxxxxxx"
        )

    with button_col:
        st.markdown("<div style='height:28px'></div>",
                    unsafe_allow_html=True)

        if manual_clicked:
            st.session_state["scan_feedback"] = None
            predict_from_barcode(manual_barcode)

            if not st.session_state.get("navigate_to_prediction"):
                st.rerun()
        

    if manual_clicked:
        st.session_state["scan_feedback"] = None
        predict_from_barcode(manual_barcode)

        # Chỉ rerun để cập nhật giao diện, không đổi trang
        if not st.session_state.get("pending_prediction"):
            st.rerun()

    st.divider()

    st.markdown("### 📸 2. Quét mã bằng hình ảnh")

    scan_col, help_col = st.columns([1.25, 0.75], gap="large")

    with scan_col:

        camera_image = st.camera_input(
            "Chụp mã vạch bằng camera",
            key="barcode_camera"
        )

        uploaded_image = st.file_uploader(
            "Hoặc tải ảnh mã vạch (PNG, JPG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            key="barcode_image_upload"
        )

        selected_image = (
            camera_image
            if camera_image is not None
            else uploaded_image
        )

        if selected_image is not None:
            st.image(
                selected_image,
                caption="Ảnh mã vạch đã chọn",
                use_container_width=True
            )

        decode_clicked = st.button(
            "🔍 Đọc mã vạch và tra cứu",
            key="decode_barcode_button",
            type="primary",
            use_container_width=True,
            disabled=selected_image is None
        )

        if decode_clicked:
            st.session_state["scan_feedback"] = None

            try:
                barcodes = decode_barcode_image(selected_image)

                if not barcodes:
                    st.session_state["scan_feedback"] = {
                        "type": "error",
                        "message": (
                            "Không nhận diện được mã vạch trong ảnh. "
                            "Hãy chụp ảnh rõ hơn, đủ sáng, tránh lóa hoặc thử nhập mã thủ công."
                        )
                    }

                else:
                    barcode = re.sub(r"\D", "", barcodes[0])

                    if len(barcode) not in (8, 12, 13, 14):
                        st.session_state["scan_feedback"] = {
                            "type": "error",
                            "message": f"Mã vạch nhận diện được không hợp lệ: {barcodes[0]}"
                        }
                    else:
                        predict_from_barcode(barcode)

            except Exception as exc:
                st.session_state["scan_feedback"] = {
                    "type": "error",
                    "message": f"Không thể đọc ảnh mã vạch: {exc}"
                }

            st.rerun()
    with help_col:

        st.markdown(
            """
            <div class="empty-box">
                <span class="emoji">📦</span>
                <b>Mẹo quét mã chính xác</b>
                Đặt mã vạch nằm thẳng, đủ sáng, không bị lóa hoặc cong nhăn.
                Ảnh nên lấy trọn vùng mã vạch và các vạch đen phải rõ nét.
            </div>
            """,
            unsafe_allow_html=True
        )

    # HIỂN THỊ MÃ VỪA QUÉT VÀ CÁC LIÊN KẾT TÌM KIẾM
    if st.session_state.get("barcode_context"):

        st.divider()

        barcode = st.session_state["barcode_context"]

        st.markdown(f"**Mã vạch gần nhất:** `{barcode}`")

        show_web_search_links(
            barcode + " thành phần sản phẩm"
        )

        if not st.session_state.get("selected_product"):

            if st.button(
                "🔮 Chuyển sang trang NOVA để nhập thành phần",
                use_container_width=True,
                type="primary"
            ):
                st.session_state["pending_prediction"] = False
                st.session_state["navigate_to_prediction"] = True
                st.rerun()

    elif not feedback:
        st.caption(
            "Chưa có mã vạch nào được quét. "
            "Bạn có thể bắt đầu bằng một trong hai cách trên."
        )
# ============================================================
# TRANG 2: TÌM SẢN PHẨM
# ============================================================
elif selected_page == "🔎 Tìm sản phẩm":
    st.markdown("## 🔎 Tìm kiếm sản phẩm thực phẩm")
    st.caption("Tìm theo tên, thương hiệu hoặc mã sản phẩm trong Open Food Facts; có liên kết để mở rộng tìm kiếm trên Internet.")

    with st.form("product_search_form", clear_on_submit=False):
        query_col, button_col = st.columns([5, 1])
        with query_col:
            product_query = st.text_input(
                "Tên sản phẩm hoặc thương hiệu",
                key="product_search_name",
                placeholder="Ví dụ: mì Hảo Hảo, Coca-Cola, sữa chua Vinamilk...",
            )
        with button_col:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Tìm kiếm", type="primary", use_container_width=True)

    if submitted:
        st.session_state["product_search_submitted"] = product_query.strip()
        st.session_state["product_search_results"] = []

    query = st.session_state.get("product_search_submitted", "")
    if query:
        show_web_search_links(query + " thực phẩm thành phần")
        st.markdown('<div class="sec-title">Kết quả từ Open Food Facts</div>', unsafe_allow_html=True)
        try:
            with st.spinner("Đang tìm sản phẩm..."):
                products = search_products_online(query)
            st.session_state["product_search_results"] = products
        except Exception as exc:  # noqa: BLE001
            products = []
            st.warning(f"Chưa thể lấy dữ liệu Open Food Facts: {exc}")

        if not products:
            st.info("Chưa tìm thấy sản phẩm phù hợp trong cơ sở dữ liệu. Bạn có thể mở các trang tìm kiếm bên trên hoặc nhập thành phần thủ công ở mục Dự đoán NOVA.")
        else:
            st.caption(f"Tìm thấy {len(products)} kết quả phù hợp. Vui lòng kiểm tra đúng tên, thương hiệu và khối lượng trước khi phân loại.")
            for index, product in enumerate(products):
                with st.container(border=True):
                    image_col, detail_col = st.columns([1, 4], gap="medium")
                    with image_col:
                        if product.get("image"):
                            st.image(product["image"], use_container_width=True)
                        else:
                            st.caption("Chưa có ảnh")
                    with detail_col:
                        st.markdown(f"**{html.escape(product['name'])}**")
                        st.caption(
                            f"Thương hiệu: {product['brand'] or 'Chưa rõ'} · "
                            f"Khối lượng: {product['quantity'] or 'Chưa rõ'} · "
                            f"Mã vạch: {product['barcode'] or 'Không có'}"
                        )
                        if product.get("ingredients"):
                            preview = product["ingredients"][:420]
                            st.write("**Thành phần:** " + preview + ("…" if len(product["ingredients"]) > 420 else ""))
                        else:
                            st.warning("Nguồn chưa cung cấp danh sách thành phần.")
                        actions = st.columns([1, 1, 2])
                        with actions[0]:
                            st.link_button("Mở nguồn", product["url"], use_container_width=True)
                        with actions[1]:
                            if st.button("Dùng sản phẩm", key=f"choose_product_{index}_{product['barcode']}", use_container_width=True):
                                classify_product(product)
                        with actions[2]:
                            if product.get("nova_group") in (1, 2, 3, 4, "1", "2", "3", "4"):
                                st.caption(f"NOVA do nguồn cung cấp: {product['nova_group']} · chỉ tham khảo")


# ============================================================
# TRANG 3: DỰ ĐOÁN NOVA
# ============================================================
elif selected_page == "🔮 Dự đoán NOVA":
    st.markdown("## 🔮 Phân loại thành phần theo NOVA")
    st.caption("Dán thành phần trên nhãn sản phẩm, hoặc chỉnh sửa nội dung lấy từ mã vạch / kết quả tìm kiếm.")

    selected_product = st.session_state.get("selected_product")
    if selected_product:
        st.info(
            f"Sản phẩm đang chọn: {selected_product.get('name', 'Không rõ tên')} · "
            f"Mã vạch: {selected_product.get('barcode') or 'Không có'}"
        )
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
            predict_clicked = st.button("Dự đoán nhóm NOVA", type="primary", use_container_width=True)
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
        # XỬ LÝ DỰ ĐOÁN ĐƯỢC CHUYỂN TỪ TRANG QUÉT MÃ / TÌM SẢN PHẨM
        if st.session_state.get("pending_prediction"):

            st.session_state["pending_prediction"] = False

            current_text = st.session_state.get("ingredients", "")
            product_for_result = st.session_state.get("selected_product")

            run_prediction_with_loader(
                current_text,
                product_for_result
            )

        # XỬ LÝ KHI NGƯỜI DÙNG NHẤN NÚT DỰ ĐOÁN THỦ CÔNG
        elif predict_clicked:

            current_text = st.session_state.get("ingredients", "")
            product_for_result = selected_product

            if (
                selected_product
                and current_text.strip()
                != selected_product.get("ingredients", "").strip()
            ):
                product_for_result = None

            run_prediction_with_loader(
                current_text,
                product_for_result
            )
        result = st.session_state.get("result")
        if result is None:
            st.markdown(
                '<div class="empty-box"><span class="emoji">🥗</span><b>Chưa có kết quả</b>'
                'Nhập thành phần ở bên trái rồi bấm Dự đoán nhóm NOVA.</div>',
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

    current_result = st.session_state.get("result")
    if current_result and current_result.get("kind") == "ok":
        render_analysis(current_result["text"])
        st.markdown('<div class="sec-title">Bốn nhóm NOVA</div>', unsafe_allow_html=True)
        render_ladder(current_result["label"])


# ============================================================
# TRANG 4: HIỂU VỀ NOVA
# ============================================================
elif selected_page == "📚 Hiểu về NOVA":
    st.markdown(
        "NOVA phân loại thực phẩm theo **mức độ và mục đích chế biến**, "
        "không dựa vào lượng dinh dưỡng. Có bốn nhóm:"
    )
    st.write("")

    for number in sorted(NOVA_META):
        m = NOVA_META[number]
        st.markdown(
            flat(
                f"""
                <div class="nv">
                  <div class="nv-bar" style="background:{m['color']}"></div>
                  <div>
                    <div class="nv-t">NOVA {number}: {m['title']}</div>
                    <p class="nv-p">{m['explain']}</p>
                    <p class="nv-e">Ví dụ: {m['examples']}</p>
                  </div>
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
elif selected_page == "🧠 Về mô hình":
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
                color="#0ea5e9",
                cornerRadiusEnd=4
            ).encode(
                x=alt.X(
                    "Điểm:Q",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(
                        title=None,
                        labelColor="#94a3b8",
                        gridColor="#334155",
                        tickColor="#475569"
                    )
                ),
                y=alt.Y(
                    "Chỉ số:N",
                    sort=None,
                    axis=alt.Axis(
                        title=None,
                        labelColor="#cbd5e1",
                        domain=False,
                        ticks=False
                    )
                ),
                tooltip=["Chỉ số", "Điểm"]
            ).properties(
                height=180,
                background="#0b1220"
            ).configure_view(
                stroke="transparent"
            ).configure_axis(
                labelFont="Inter",
                labelFontSize=12
            )

            st.altair_chart(chart, use_container_width=True)

            with text_col:
                st.markdown("""
    <div style="
    padding:16px;
    border-radius:16px;
    background:rgba(6,182,212,.10);
    border:1px solid rgba(6,182,212,.25);
    color:#e2e8f0;
    line-height:1.6;
    ">
    📊 Mô hình Linear SVC đạt Macro-F1 87,96%.
    NOVA 2 là nhóm có F1-score thấp nhất do số lượng
    mẫu ít hơn đáng kể so với các nhóm còn lại.
    </div>
    """, unsafe_allow_html=True)

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
        "- Phân bố nhãn không đồng đều, NOVA 2 có rất ít mẫu.\n"
        "- Mô hình chỉ đọc văn bản nên chưa hiểu quy trình chế biến thực tế.\n"
        "- Nhãn trong Open Food Facts do cộng đồng đóng góp, có thể có sai lệch."
    )



# ============================================================
# CHÂN TRANG
# ============================================================
st.divider()
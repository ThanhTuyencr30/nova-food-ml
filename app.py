import html
import re
import time

import pandas as pd
import streamlit as st
import altair as alt

from src.nova.inference.predict import (
    predict_ingredients,
    predict_proba_ingredients,
)

# ==============================
# CẤU HÌNH TRANG
# ==============================
st.set_page_config(
    page_title="NOVA Food Classifier",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

MIN_LOADING_SECONDS = 2.2  # thời gian tối thiểu hiển thị hiệu ứng chờ

# ==============================
# DỮ LIỆU NỘI DUNG
# ==============================
NOVA_META = {
    1: {
        "title": "Chưa hoặc ít chế biến",
        "color": "#22c55e",
        "explain": "Thực phẩm tự nhiên hoặc chỉ qua xử lý tối thiểu như rửa, sấy, xay, đông lạnh, không thêm chất phụ gia.",
        "examples": "Rau củ, trái cây, thịt cá tươi, trứng, sữa tươi, gạo, đậu.",
    },
    2: {
        "title": "Nguyên liệu chế biến",
        "color": "#45beea",
        "explain": "Chất được chiết xuất từ nhóm 1 hoặc từ thiên nhiên, dùng để nêm nếm và nấu nướng, ít khi ăn riêng.",
        "examples": "Dầu ăn, bơ, đường, muối, mật ong, giấm.",
    },
    3: {
        "title": "Thực phẩm chế biến",
        "color": "#f97316",
        "explain": "Kết hợp thực phẩm nhóm 1 với nguyên liệu nhóm 2 (muối, đường, dầu) để bảo quản hoặc tăng hương vị.",
        "examples": "Bánh mì thủ công, phô mai, cá hộp, rau củ muối, trái cây ngâm.",
    },
    4: {
        "title": "Siêu chế biến",
        "color": "#ea1c1c",
        "explain": "Công thức công nghiệp có nhiều thành phần mà bếp nhà ít dùng như phụ gia, hương liệu, chất tạo màu, nhũ hóa.",
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

# Quét từ khóa để minh họa, KHÔNG dùng để dự đoán
MARKERS = {
    "Mã phụ gia E": r"\be\s?\d{3,4}[a-z]?\b",
    "Hương liệu": r"flavou?r",
    "Chất tạo màu": r"colou?r|caramel",
    "Chất nhũ hóa": r"emulsifier|lecithin|diglycerides",
    "Chất làm ngọt": r"sweetener|aspartame|sucralose|acesulfame|glucose[- ]fructose|fructose syrup|glucose syrup",
    "Maltodextrin / tinh bột biến tính": r"maltodextrin|modified starch|dextrose",
    "Chất điều vị": r"glutamate|inosinate|flavou?r enhancer",
    "Chất bảo quản / điều chỉnh độ acid": r"preservative|acidity regulator|benzoate|sorbate|citric acid",
}

DEFAULT_META = {"title": "", "color": "#FFFFFF", "explain": "", "examples": ""}


def meta(label: int) -> dict:
    return NOVA_META.get(label, DEFAULT_META)


def flat(s: str) -> str:
    """Nén HTML về một dòng để Markdown không coi là code block."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


# ==============================
# CSS
# ==============================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
    --primary:#06b6d4;
    --secondary:#34d399;
    --bg:#050816;
    --card:#111827;
    --border:rgba(255,255,255,.08);
    --text:#f8fafc;
    --muted:#94a3b8;
}

html,body,.stApp{
    background:
        radial-gradient(circle at top right,#06b6d433,transparent 25%),
        linear-gradient(180deg,#050816,#0b1220);
    color:var(--text);
    font-family:'Inter',sans-serif;
}

.block-container{ max-width:1300px; padding-top:1.5rem; }
footer{ visibility:hidden; }

/* HERO */
.hero{
    position:relative; overflow:hidden; padding:42px; border-radius:28px;
    background:linear-gradient(135deg,rgba(6,182,212,.15),rgba(52,211,153,.08));
    border:1px solid rgba(255,255,255,.08);
    backdrop-filter:blur(18px);
    box-shadow:0 20px 50px rgba(0,0,0,.35);
    margin-bottom:1rem;
}
.hero::after{
    content:""; position:absolute; width:380px; height:380px;
    right:-120px; top:-120px; border-radius:50%;
    background:radial-gradient(circle,rgba(6,182,212,.25),transparent 70%);
}
.hero-tag{ color:#67e8f9; font-size:.85rem; font-weight:600; letter-spacing:1px; }
.hero-title{ font-size:3rem; font-weight:800; margin-top:10px; }
.hero-sub{ margin-top:10px; color:#cbd5e1; max-width:700px; }
.pill{
    display:inline-block; margin-top:18px; margin-right:8px; padding:8px 14px;
    border-radius:999px; background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.08);
}

/* SECTION TITLE */
.sec-title{ font-size:1.25rem; font-weight:700; margin:28px 0 12px; }

/* EMPTY BOX */
.empty-box{
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:6px; min-height:300px; text-align:center; padding:24px;
    border:1px dashed rgba(255,255,255,.15); border-radius:22px; color:var(--muted);
}
.empty-box span{ font-size:2.6rem; }
.empty-box b{ color:var(--text); font-size:1.1rem; }

/* RESULT CARD */
.prediction-card{
    border-radius:24px; padding:28px;
    background:linear-gradient(145deg,rgba(255,255,255,.05),rgba(255,255,255,.03));
    border:1px solid rgba(255,255,255,.08);
}
.prediction-badge{
    display:inline-flex; align-items:center; gap:10px;
    padding:14px 24px; border-radius:999px; font-size:2rem; font-weight:800;
}
.prediction-title{ margin-top:20px; font-size:1.4rem; font-weight:700; }
.prediction-desc{ margin-top:8px; color:#cbd5e1; margin-bottom:10px; }

/* Probability */
.score-row{ margin-top:14px; }
.score-head{ display:flex; justify-content:space-between; font-size:.9rem; margin-bottom:5px; }
.score-track{ height:12px; background:#1e293b; border-radius:999px; overflow:hidden; }
.score-fill{ height:100%; border-radius:999px; animation:grow .8s ease; }
@keyframes grow{ from{ width:0; } }

/* NOVA CARD */
.nova-card{
    height:100%; border-radius:18px; padding:20px;
    background:#111827; border-top:5px solid;
}
.nova-number{ font-size:2rem; font-weight:800; }
.nova-title{ font-weight:700; margin-top:6px; }
.nova-example{ margin-top:10px; color:#94a3b8; font-size:.88rem; }

/* CHIP */
.chip{
    display:inline-block; padding:7px 12px; background:#1e293b;
    border-radius:999px; margin:4px; font-size:.85rem;
}
.chip.flag{ background:#ef444420; color:#ef4444; border:1px solid #ef444455; }

/* TAB NOVA */
.nv{
    display:flex; gap:16px; padding:18px; margin-bottom:12px;
    background:rgba(255,255,255,.03); border:1px solid var(--border); border-radius:18px;
}
.nv-bar{ width:6px; border-radius:999px; flex-shrink:0; }
.nv-t{ font-weight:700; font-size:1.1rem; }
.nv-p{ margin:6px 0 0; color:#cbd5e1; }
.nv-e{ margin:6px 0 0; color:var(--muted); font-size:.9rem; }

/* TAB MODEL */
.step{
    height:100%; padding:20px; border-radius:18px;
    background:rgba(255,255,255,.03); border:1px solid var(--border);
}
.step-n{
    width:34px; height:34px; border-radius:50%; display:flex;
    align-items:center; justify-content:center; font-weight:800; color:#000;
    background:linear-gradient(90deg,#06b6d4,#34d399);
}
.step-t{ font-weight:700; margin-top:12px; }
.step-d{ color:var(--muted); font-size:.9rem; margin-top:4px; }

/* SIDEBAR */
[data-testid="stSidebar"]{ background:#09111f; }
.hist{
    display:flex; align-items:center; gap:8px; padding:8px 0;
    font-size:.85rem; border-bottom:1px solid var(--border);
}
.hist-dot{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.hist-txt{
    color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}

/* BUTTON */
.stButton button{
    border:none !important; border-radius:14px !important;
    background:linear-gradient(90deg,#06b6d4,#34d399) !important;
    color:black !important; font-weight:700 !important; transition:.25s;
}
.stButton button:hover{ transform:translateY(-2px); }

/* TEXT AREA */
textarea{
    border-radius:18px !important;
    background:#0f172a !important;

    border:1px solid rgba(6,182,212,.35) !important;

    box-shadow:
        0 0 8px rgba(6,182,212,.20);
}

textarea:focus{
    border:1px solid #00ffff !important;

    box-shadow:
        0 0 10px #00ffff,
        0 0 20px #00ffff,
        0 0 40px rgba(0,255,255,.5);

    outline:none !important;
}
.stTabs [aria-selected="true"]{
    border-bottom:2px solid #00ffff !important;

    text-shadow:
        0 0 8px #00ffff;
}
/* LOADER */
.ld{
    display:flex; flex-direction:column; align-items:center; gap:18px;
    min-height:300px; justify-content:center; padding:24px;
    border-radius:22px; background:rgba(255,255,255,.03); border:1px solid var(--border);
}
.ld-top{ position:relative; width:90px; height:90px; }
.ld-ring{
    position:absolute; inset:0; border-radius:50%;
    border:4px solid transparent; border-top-color:#06b6d4;
    animation:spin 1s linear infinite;
}
.ld-ring.b{ inset:10px; border-top-color:#34d399; animation-direction:reverse; }
.ld-core{
    position:absolute; inset:0; display:flex; align-items:center;
    justify-content:center; font-size:1.8rem;
}
@keyframes spin{ to{ transform:rotate(360deg); } }

.ld-msgs{ position:relative; height:24px; width:100%; text-align:center; color:#cbd5e1; }
.ld-msgs span{ position:absolute; left:0; right:0; opacity:0; animation:msg 2.2s forwards; }
.ld-msgs span:nth-child(2){ animation-delay:.73s; }
.ld-msgs span:nth-child(3){ animation-delay:1.46s; }
@keyframes msg{
    0%{ opacity:0; } 6%{ opacity:1; } 30%{ opacity:1; } 36%{ opacity:0; } 100%{ opacity:0; }
}
.ld-msgs span:nth-child(3){ animation-name:msg-last; }
@keyframes msg-last{ 0%{ opacity:0; } 6%{ opacity:1; } 100%{ opacity:1; } }

.ld-track{ position:relative; width:80%; height:28px; }
.ld-runner{ position:absolute; left:0; top:0; animation:run 2.2s linear forwards; }
@keyframes run{ to{ left:calc(100% - 24px); } }
.ld-bar{ width:80%; height:6px; background:#1e293b; border-radius:999px; overflow:hidden; }
.ld-bar i{
    display:block; height:100%; width:0;
    background:linear-gradient(90deg,#06b6d4,#34d399);
    animation:fill 2.2s linear forwards;
}
@keyframes fill{ to{ width:100%; } }
/* ÉP MÀU CHỮ SÁNG */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stMarkdown, [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] p,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"]{
    color:#e2e8f0 !important;
}

[data-testid="stCaptionContainer"]{ color:#94a3b8 !important;
opacity:.9; font-size:.85rem; }

/* Thanh trên cùng */
[data-testid="stHeader"]{ background:transparent !important; }

/* Sidebar */
[data-testid="stSidebar"] *{ color:#e2e8f0; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *{ color:#94a3b8; }

/* Tab */
.stTabs [data-baseweb="tab"] p{ color:#94a3b8 !important; font-size:1rem !important; }
.stTabs [aria-selected="true"] p{ color:#67e8f9 !important; }

/* Ô nhập */
textarea{ color:#f8fafc !important; }
textarea::placeholder{ color:#64748b !important; }

/* Expander (đang bị nền trắng) */
[data-testid="stExpander"] details{
    background:rgba(255,255,255,.03) !important;
    border:1px solid rgba(255,255,255,.08) !important;
    border-radius:16px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *{
    background:transparent !important;
    color:#e2e8f0 !important;
}

/* DATAFRAME DARK MODE */
[data-testid="stDataFrame"] {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 16px !important;
}

[data-testid="stDataFrame"] table {
    background: #111827 !important;
}

[data-testid="stDataFrame"] th {
    background: #1e293b !important;
    color: #67e8f9 !important;
    font-weight: 700 !important;
}

[data-testid="stDataFrame"] td {
    background: #111827 !important;
    color: #e2e8f0 !important;
}

[data-testid="stDataFrame"] tr:hover td {
    background: #172033 !important;
}

/* Bảng */
[data-testid="stTable"] *{ color:#e2e8f0 !important; }
</style>
""",
    unsafe_allow_html=True,
    
)


# ==============================
# TRẠNG THÁI & HÀM XỬ LÝ
# ==============================
st.session_state.setdefault("ingredients", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("pending", False)


def request_prediction() -> None:
    st.session_state["pending"] = True


def use_example(text: str) -> None:
    st.session_state["ingredients"] = text
    st.session_state["result"] = None


def clear_all() -> None:
    st.session_state["ingredients"] = ""
    st.session_state["result"] = None


def clear_history() -> None:
    st.session_state["history"] = []


def run_prediction() -> None:
    text = st.session_state["ingredients"].strip()

    if not text:
        st.session_state["result"] = {
            "kind": "warning",
            "message": "Bạn chưa nhập thành phần. Hãy dán danh sách thành phần từ nhãn sản phẩm hoặc chọn một ví dụ.",
        }
        return

    try:
        label, description = predict_ingredients(text)
        raw_scores = predict_proba_ingredients(text)
        scores = {int(k): float(v) for k, v in raw_scores.items()}
        st.session_state["result"] = {
            "kind": "ok",
            "text": text,
            "label": int(label),
            "description": description,
            "scores": scores,
        }
        st.session_state["history"].insert(0, {"label": int(label), "text": text})
        st.session_state["history"] = st.session_state["history"][:6]
    except FileNotFoundError as exc:
        st.session_state["result"] = {
            "kind": "error",
            "message": f"Chưa tìm thấy mô hình đã huấn luyện. Hãy chạy bước huấn luyện trước. Chi tiết: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        st.session_state["result"] = {
            "kind": "error",
            "message": f"Không thể dự đoán lúc này. Chi tiết: {exc}",
        }


def split_ingredients(text: str) -> list[str]:
    parts = re.split(r"[,;]", text)
    return [p.strip(" .\n\t") for p in parts if p.strip(" .\n\t")]


def scan_markers(text: str) -> dict:
    low = text.lower()
    found = {}
    for name, pattern in MARKERS.items():
        n = len(re.findall(pattern, low))
        if n:
            found[name] = n
    return found


def render_loader() -> str:
    return flat(
        """
        <div class="ld">
          <div class="ld-top">
            <div class="ld-ring"></div><div class="ld-ring b"></div>
            <div class="ld-core">🥗</div>
          </div>
          <div class="ld-msgs">
            <span>Chuẩn hóa văn bản thành phần...</span>
            <span>Vector hóa bằng TF-IDF...</span>
            <span>Tính điểm bốn nhóm NOVA...</span>
          </div>
          <div class="ld-track"><span class="ld-runner">🐇</span></div>
          <div class="ld-bar"><i></i></div>
        </div>
        """
    )


def render_label_panel(result: dict) -> str:
    label = result["label"]
    m = meta(label)

    rows = '<div class="score-title">Điểm số tương đối theo từng nhóm NOVA (softmax, không phải xác suất)</div>'

    for k, v in sorted(result["scores"].items()):
        rows += (
            '<div class="score-row">'
            f'<div class="score-head"><span>NOVA {k}</span><span>{v * 100:.1f} điểm</span></div>'
            '<div class="score-track">'
            f'<div class="score-fill" style="width:{v * 100:.1f}%;background:{meta(k)["color"]};"></div>'
            "</div></div>"
        )

    return flat(
        f"""
        <div class="prediction-card">
          <div class="prediction-badge" style="background:{m['color']}22;color:{m['color']};">
            🔥 NOVA {label}
          </div>
          <div class="prediction-title">{m['title']}</div>
          <div class="prediction-desc">{m['explain']}</div>
          {rows}
        </div>
        """
    )


def render_ladder(active=None) -> None:
    cols = st.columns(4)
    for col, n in zip(cols, [1, 2, 3, 4]):
        m = NOVA_META[n]
        glow = f"box-shadow:0 0 25px {m['color']}55;" if n == active else ""
        col.markdown(
            flat(
                f"""
                <div class="nova-card" style="border-top-color:{m['color']};{glow}">
                  <div class="nova-number">{n}</div>
                  <div class="nova-title">{m['title']}</div>
                  <div class="nova-example">{m['examples']}</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )


# ==============================
# HERO
# ==============================
st.markdown(
    flat(
        """
        <div class="hero">
          <div class="hero-tag">🥗 NOVA.CLASSIFIER</div>
          <div class="hero-title">Ăn uống thông minh bắt đầu từ việc hiểu thực phẩm!</div>
          <div class="hero-sub">
            Khám phá những gì ẩn sau danh sách thành phần trên bao bì cùng NOVA.CLASSIFIER.
          </div>
          <div>
            <span class="pill">4 nhóm NOVA</span>
            <span class="pill">TF-IDF</span>
            <span class="pill">Linear SVC</span>
            <span class="pill">95.76% Accuracy</span>
          </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# ==============================
# TABS  (trước đây bị thiếu đoạn này)
# ==============================
tab_predict, tab_nova, tab_model = st.tabs(
    ["🔮 Dự đoán", "📚 Hiểu về NOVA", "🧠 Về mô hình"]
)

# ==============================
# TAB 1: DỰ ĐOÁN
# ==============================
with tab_predict:
    left, right = st.columns([5, 6], gap="large")

    with left:
        st.text_area(
            "Thành phần trên nhãn",
            key="ingredients",
            height=200,
            placeholder="Ví dụ: sugar, wheat flour, vegetable oil, salt",
            help="Ngăn cách các thành phần bằng dấu phẩy, giữ nguyên thứ tự như trên nhãn.",
        )

        b1, b2 = st.columns([2, 1])
        with b1:
            st.button("Dự đoán nhóm NOVA", type="primary", use_container_width=True, on_click=request_prediction)
        with b2:
            st.button("Xóa", use_container_width=True, on_click=clear_all)

        st.markdown('<div class="sec-title">Thử nhanh với một ví dụ</div>', unsafe_allow_html=True)
        ex_cols = st.columns(len(EXAMPLES))
        for col, (name, text) in zip(ex_cols, EXAMPLES.items()):
            with col:
                st.button(name, key=f"ex_{name}", use_container_width=True, on_click=use_example, args=(text,))

        with st.expander("Mẹo nhập để kết quả tốt hơn"):
            st.markdown(
                "- Nhập đủ toàn bộ danh sách thành phần, đừng chỉ chép vài dòng đầu.\n"
                "- Giữ nguyên thứ tự trên nhãn, vì thành phần được xếp theo khối lượng giảm dần.\n"
                "- Các ví dụ dùng tiếng Anh. Nếu nhãn tiếng Việt cho kết quả lạ, hãy thử nhập bản tiếng Anh."
            )

    with right:
        slot = st.empty()

        if st.session_state["pending"]:
            st.session_state["pending"] = False
            if st.session_state["ingredients"].strip() and st.session_state.get("fx", True):
                slot.markdown(render_loader(), unsafe_allow_html=True)
                started = time.time()
                run_prediction()
                remaining = MIN_LOADING_SECONDS - (time.time() - started)
                if remaining > 0:
                    time.sleep(remaining)
            else:
                run_prediction()

        result = st.session_state["result"]

        with slot.container():
            if result is None:
                st.markdown(
                    '<div class="empty-box"><span>🥗</span><b>Chưa có kết quả</b>'
                    "Nhập thành phần rồi bấm Dự đoán, hoặc chọn một ví dụ bên trái.</div>",
                    unsafe_allow_html=True,
                )
            elif result["kind"] == "warning":
                st.warning(result["message"])
            elif result["kind"] == "error":
                st.error(result["message"])
            else:
                st.markdown(render_label_panel(result), unsafe_allow_html=True)

    # Phân tích thành phần (chỉ khi có kết quả)
    if result and result["kind"] == "ok":
        items = split_ingredients(result["text"])
        found = scan_markers(result["text"])

        st.markdown('<div class="sec-title">Phân tích thành phần</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Số thành phần", len(items))
        m2.metric("Loại dấu hiệu phụ gia", len(found))
        m3.metric("Tổng lần xuất hiện", sum(found.values()))

        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("**Thành phần đã đọc**")
            chips = "".join(f'<span class="chip">{html.escape(i)}</span>' for i in items[:40])
            if len(items) > 40:
                chips += f'<span class="chip">và {len(items) - 40} thành phần khác</span>'
            st.markdown(chips, unsafe_allow_html=True)
            if len(items) < 3:
                st.caption("Danh sách khá ngắn nên độ tin cậy có thể thấp.")
        with b:
            st.markdown("**Dấu hiệu thường gặp ở thực phẩm siêu chế biến**")
            if found:
                flags = "".join(
                    f'<span class="chip flag">{html.escape(k)} x{v}</span>' for k, v in found.items()
                )
                st.markdown(flags, unsafe_allow_html=True)
            else:
                st.caption("Không thấy phụ gia hay chất tạo hương, màu, nhũ hóa phổ biến trong danh sách.")
            st.caption("Phần này chỉ quét từ khóa để minh họa, không dùng để tính kết quả dự đoán.")

    st.markdown('<div class="sec-title">Bốn nhóm NOVA</div>', unsafe_allow_html=True)
    render_ladder(result["label"] if result and result["kind"] == "ok" else None)

# ==============================
# TAB 2: HIỂU VỀ NOVA
# ==============================
with tab_nova:
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

# ==============================
# TAB 3: VỀ MÔ HÌNH
# ==============================
with tab_model:
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

st.divider()
st.caption("Đồ án học thuật - NOVA Food Processing Classification")

# ==============================
# SIDEBAR (đặt cuối để lịch sử luôn mới nhất)
# ==============================
with st.sidebar:
    st.markdown("### 🥗 NOVA Classifier")
    st.caption("Phân loại thực phẩm theo mức độ chế biến")

    st.markdown("**Cách dùng**")
    st.markdown(
        "1. Dán danh sách thành phần từ nhãn.\n"
        "2. Bấm **Dự đoán nhóm NOVA**.\n"
        "3. Xem nhóm dự đoán, điểm số tương đối và các dấu hiệu phụ gia."
    )
    st.toggle("Hiệu ứng khi chờ kết quả", value=True, key="fx")

    st.divider()
    st.markdown("**Lịch sử gần đây**")
    history = st.session_state["history"]
    if not history:
        st.caption("Chưa có dự đoán nào trong phiên này.")
    else:
        for h in history:
            snippet = html.escape(h["text"][:60])
            st.markdown(
                f'<div class="hist"><span class="hist-dot" style="background:{meta(h["label"])["color"]}"></span>'
                f'<b>NOVA {h["label"]}</b><span class="hist-txt">{snippet}</span></div>',
                unsafe_allow_html=True,
            )
        st.button("Xóa lịch sử", on_click=clear_history, use_container_width=True)

    st.divider()
st.caption("Công nghệ: Python, Streamlit, TF-IDF, Linear SVC.")
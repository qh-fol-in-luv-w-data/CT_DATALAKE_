import streamlit as st

from apps.ct_datalake.ct_datalake.search import search
from apps.ct_datalake.ct_datalake.llm_rerank import rerank

from apps.ct_datalake.ct_datalake.jd_match import (
    render_jd_uploader,
    match_jd,
    render_match_results,
    normalize_candidate,
)

from apps.ct_datalake.ct_datalake.ai_matching  import (
    extract_keywords,
    search_domain,
    load_datasets,
)
import apps.ct_datalake.ct_datalake.ai_matching as _rc
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
api_key = os.getenv("OPENAI_API_KEY", "").strip()
gpt_client = OpenAI(api_key=api_key) if api_key else None
# ========================
# CONFIG
# ========================
st.set_page_config(
    page_title="Candidate Search",
    layout="wide"
)

# ========================
# STYLE
# ========================
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 18px;
    background: rgba(255,255,255,0.03);
}

.metric-box {
    padding: 12px;
    border-radius: 12px;
    background: rgba(255,255,255,0.04);
    text-align: center;
}

.small-label {
    font-size: 12px;
    opacity: 0.7;
}

.big-value {
    font-size: 22px;
    font-weight: 700;
}

.tag {
    display: inline-block;
    padding: 4px 10px;
    margin: 4px 4px 0 0;
    border-radius: 999px;
    background: rgba(0,123,255,0.15);
    font-size: 13px;
}

.result-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
}

.subtle {
    opacity: 0.8;
    font-size: 14px;
}

.domain-header {
    font-size: 18px;
    font-weight: 700;
    padding: 10px 0 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 12px;
}

.kw-badge {
    display: inline-block;
    padding: 3px 9px;
    margin: 3px 3px 0 0;
    border-radius: 999px;
    background: rgba(255,165,0,0.15);
    font-size: 12px;
    color: #ffaa44;
}

</style>
""", unsafe_allow_html=True)

# ========================
# HEADER
# ========================
st.title("🔍 AI Candidate Search")
st.caption("Semantic Search + JD Matching + LLM Ranking")

# =========================================================
# SEARCH
# =========================================================
st.divider()

st.header("🔎 Semantic Search")

col_q1, col_q2 = st.columns([4, 1])

with col_q1:
    query = st.text_input(
        "Nhập yêu cầu",
        placeholder="Ví dụ: AI engineer, NLP, fintech, professor robotics..."
    )

with col_q2:
    top_k = st.number_input(
        "Top K",
        min_value=1,
        max_value=20,
        value=5
    )

col_mode1, col_mode2 = st.columns(2)

with col_mode1:
    dataset_mode = st.radio(
        "Nguồn dữ liệu",
        [
            "🏢 Nội bộ",
            "🌐 Bên ngoài"
        ]
    )

with col_mode2:
    search_type = st.radio(
        "Chế độ",
        [
            "⚡ FAISS",
            "🧠 LLM"
        ]
    )

search_mode = (
    "in"
    if dataset_mode == "🏢 Nội bộ"
    else "out"
)

# ========================
# SEARCH BUTTON
# ========================
if st.button("🚀 Search", use_container_width=True):

    if not query.strip():
        st.warning("Vui lòng nhập nội dung tìm kiếm")
        st.stop()

    with st.spinner("Đang tìm kiếm ứng viên..."):

        raw_results = search(
            query=query,
            mode=search_mode,
            top_k=max(top_k * 2, 10)
        )

    if not raw_results:
        st.warning("Không tìm thấy kết quả phù hợp")
        st.stop()

    # =====================================================
    # FAISS MODE
    # =====================================================
    if search_type == "⚡ FAISS":

        st.success(f"✅ Tìm thấy {len(raw_results)} ứng viên")

        for i, r in enumerate(raw_results[:top_k], 1):

            d = normalize_candidate(
                r["data"],
                search_mode
            )

            score = round(r["score"], 4)

            with st.container():

                st.markdown('<div class="card">', unsafe_allow_html=True)

                # HEADER
                col1, col2 = st.columns([5, 1])

                with col1:

                    st.markdown(
                        f"""
                        <div class="result-title">
                            #{i} {d['name']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:

                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="small-label">FAISS SCORE</div>
                            <div class="big-value">{score}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # =========================================
                # INTERNAL
                # =========================================
                if search_mode == "in":

                    c1, c2 = st.columns(2)

                    with c1:

                        if d["school"]:
                            st.markdown(
                                f"🏫 **Trường:** {d['school']}"
                            )

                        if d["academic_rank"] or d["degree"]:
                            st.markdown(
                                f"🎓 **Học hàm/vị:** "
                                f"{d['academic_rank']} "
                                f"{d['degree']}"
                            )

                    with c2:

                        if d["position"]:
                            st.markdown(
                                f"💼 **Chức vụ:** {d['position']}"
                            )

                    if d["expertise"]:

                        st.markdown("### 🧠 Mô tả nghiên cứu / sản phẩm")

                        st.info(d["expertise"])

                # =========================================
                # EXTERNAL
                # =========================================
                else:

                    c1, c2 = st.columns(2)

                    with c1:

                        if d["affiliation"]:
                            st.markdown(
                                f"🏢 **Affiliation:** "
                                f"{d['affiliation']}"
                            )

                        if d["university_name"]:
                            st.markdown(
                                f"🎓 **University:** "
                                f"{d['university_name']} "
                                f"({d['university_abbr']})"
                            )

                        if d["city"]:
                            st.markdown(
                                f"📍 **City:** {d['city']}"
                            )

                    with c2:

                        if d["email"]:
                            st.markdown(
                                f"📧 **Email:** {d['email']}"
                            )

                        st.markdown(
                            f"📚 **Citations:** {d['citations']}"
                        )

                        st.markdown(
                            f"📈 **h-index:** {d['h_index']}"
                        )

                    if d["interests"]:

                        st.markdown("### 🔬 Research Interests")

                        tag_html = ""

                        for t in d["interests"]:
                            tag_html += (
                                f'<span class="tag">{t}</span>'
                            )

                        st.markdown(
                            tag_html,
                            unsafe_allow_html=True
                        )

                    if d["url"]:

                        st.link_button(
                            "🔗 Google Scholar",
                            d["url"],
                            use_container_width=True
                        )

                st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # LLM MODE
    # =====================================================
    else:

        with st.spinner("LLM đang rerank ứng viên..."):

            answer = rerank(
                query=query,
                candidates=raw_results[:top_k]
            )

        st.markdown("## 🧠 Phân tích LLM")

        st.write(answer)

# =========================================================
# JD MATCHING
# =========================================================
st.divider()

st.header("📄 JD Matching")

col_j1, col_j2 = st.columns(2)

with col_j1:

    jd_dataset_mode = st.radio(
        "Nguồn dữ liệu JD",
        [
            "🏢 Nội bộ",
            "🌐 Bên ngoài"
        ],
        key="jd_dataset"
    )

with col_j2:

    jd_top_k = st.slider(
        "Top ứng viên",
        1,
        20,
        5,
        key="jd_top_k"
    )

jd_mode = (
    "in"
    if jd_dataset_mode == "🏢 Nội bộ"
    else "out"
)

jd_fast = st.checkbox(
    "⚡ Fast mode (không rerank LLM)",
    value=False
)

# ========================
# JD INPUT
# ========================
jd_text = render_jd_uploader()

# ========================
# MATCH BUTTON
# ========================
if st.button(
    "🚀 Match JD",
    use_container_width=True
):

    if not jd_text or not jd_text.strip():
        st.warning("Vui lòng upload hoặc nhập JD")
        st.stop()

    with st.spinner(
        "Đang phân tích JD và matching ứng viên..."
    ):

        results = match_jd(
            jd_text=jd_text,
            mode=jd_mode,
            top_k=jd_top_k,
            fast=jd_fast,
        )

    # dùng render đẹp từ jd_match.py
    render_match_results(
        results=results,
        mode=jd_mode
    )

# =========================================================
# G600 – TỜ TRÌNH
# =========================================================
st.divider()

st.header("📋 Đề xuất ứng viên từ Tờ trình G600")
st.caption("Upload PDF tờ trình → GPT trích xuất lĩnh vực → FAISS tìm ứng viên phù hợp")

# --- Upload PDF ---
g600_pdf = st.file_uploader(
    "📎 Upload tờ trình PDF",
    type=["pdf"],
    key="g600_pdf"
)

# --- Options row (ngay trên nút Run) ---
opt_col1, opt_col2 = st.columns(2)

with opt_col1:
    g600_top_k = st.slider(
        "Top K mỗi lĩnh vực",
        min_value=1,
        max_value=10,
        value=5,
        key="g600_top_k"
    )

with opt_col2:
    g600_source = st.radio(
        "Nguồn dữ liệu",
        ["🏢 Nội bộ", "🌐 Bên ngoài", "🔀 Cả hai"],
        key="g600_source"
    )

# --- Run button ---
if st.button("🚀 Phân tích & Đề xuất ứng viên", use_container_width=True, key="g600_run"):

    # Validate
    if not g600_pdf:
        st.warning("Vui lòng upload file PDF tờ trình")
        st.stop()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.error("Không tìm thấy OPENAI_API_KEY trong môi trường. Vui lòng set biến môi trường trước khi chạy app.")
        st.stop()

    # --- Step 1+2: Vision đọc PDF & trích keywords cùng lúc ---
    with st.spinner("🤖 GPT-4o Vision đang đọc tờ trình và trích xuất keywords..."):
        import tempfile, os as _os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(g600_pdf.read())
            tmp_path = tmp.name

        try:
            domains = extract_keywords(tmp_path, gpt_client)
        finally:
            _os.unlink(tmp_path)

    if not domains:
        st.error("GPT không trích xuất được lĩnh vực nào. Kiểm tra lại API key hoặc nội dung PDF.")
        st.stop()

    st.success(f"✅ Tìm thấy **{len(domains)} lĩnh vực** cần tuyển chuyên gia")

    # --- Step 3: Load FAISS & search ---
    with st.spinner("⏳ Load FAISS datasets..."):

        @st.cache_resource
        def _load_datasets():
            return load_datasets()

        @st.cache_resource
        def _load_embed():
            return SentenceTransformer("intfloat/multilingual-e5-base")

        datasets   = _load_datasets()
        embed_model = _load_embed()

    # Filter mode
    g600_src = g600_source
    if g600_src == "🏢 Nội bộ":
        active_datasets = {k: v for k, v in datasets.items() if k == "in"}
    elif g600_src == "🌐 Bên ngoài":
        active_datasets = {k: v for k, v in datasets.items() if k == "out"}
    else:
        active_datasets = datasets

    if not active_datasets:
        st.error("Không tìm thấy FAISS index. Kiểm tra lại file index.faiss / index_out.faiss")
        st.stop()

    top_k_val = g600_top_k

    # Patch TOP_K
    _rc.TOP_K           = top_k_val
    _rc.SCORE_THRESHOLD = 0.40

    # --- Step 4: Render từng lĩnh vực ---
    st.divider()
    st.subheader("🎯 Kết quả đề xuất ứng viên")

    for domain in domains:

        with st.expander(f"🔬 {domain['name']}", expanded=True):

            # Keywords badges
            kw_html = ""
            for kw in domain.get("keywords_en", [])[:6]:
                kw_html += f'<span class="kw-badge">{kw}</span>'
            for kw in domain.get("keywords_vi", [])[:4]:
                kw_html += f'<span class="tag">{kw}</span>'

            st.markdown(kw_html, unsafe_allow_html=True)
            st.caption(f"Query VI: `{domain['query_vi']}`")
            st.caption(f"Query EN: `{domain['query_en']}`")

            with st.spinner(f"Đang tìm ứng viên cho {domain['name']}..."):
                hits = search_domain(domain, active_datasets, embed_model)

            if not hits:
                st.warning("Không tìm thấy ứng viên phù hợp (score thấp hơn ngưỡng)")
                continue

            st.success(f"✅ Tìm thấy {len(hits)} ứng viên")

            for i, hit in enumerate(hits, 1):

                d    = hit["data"]
                mode = hit["mode"]
                src_label = "🌐 Google Scholar" if mode == "out" else "🏫 Nội bộ"

                st.markdown('<div class="card">', unsafe_allow_html=True)

                hdr_col, score_col = st.columns([5, 1])

                with hdr_col:
                    name = d.get("name") if mode == "out" else (d.get("trường họ và tên") or d.get("name") or "")
                    st.markdown(
                        f'<div class="result-title">#{i} {name} &nbsp;<span class="subtle">{src_label}</span></div>',
                        unsafe_allow_html=True
                    )

                with score_col:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="small-label">SCORE</div>
                            <div class="big-value">{hit['score']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # --- External (Google Scholar) ---
                if mode == "out":

                    ec1, ec2 = st.columns(2)

                    with ec1:
                        if d.get("affiliation"):
                            st.markdown(f"🏢 **Affiliation:** {d['affiliation']}")
                        if d.get("university_name"):
                            st.markdown(f"🎓 **University:** {d['university_name']} ({d.get('university_abbr', '')})")
                        if d.get("city"):
                            st.markdown(f"📍 **City:** {d['city']}")

                    with ec2:
                        if d.get("email"):
                            st.markdown(f"📧 **Email:** {d['email']}")
                        st.markdown(f"📚 **Citations:** {d.get('citations', 'N/A')}")
                        st.markdown(f"📈 **h-index:** {d.get('h_index', 'N/A')}")

                    if d.get("interests"):
                        st.markdown("**🔬 Research Interests:**")
                        tag_html = "".join(
                            f'<span class="tag">{t}</span>'
                            for t in d["interests"]
                        )
                        st.markdown(tag_html, unsafe_allow_html=True)

                    if d.get("url"):
                        st.link_button("🔗 Google Scholar Profile", d["url"], use_container_width=True)

                # --- Internal ---
                else:

                    ic1, ic2 = st.columns(2)

                    with ic1:
                        if d.get("trường") or d.get("affiliation"):
                            st.markdown(f"🏫 **Trường:** {d.get('trường') or d.get('affiliation')}")
                        if d.get("học hàm") or d.get("học vị"):
                            st.markdown(f"🎓 **Học hàm/vị:** {d.get('học hàm', '')} {d.get('học vị', '')}")

                    with ic2:
                        if d.get("chức vụ"):
                            st.markdown(f"💼 **Chức vụ:** {d['chức vụ']}")

                    sp = d.get("sản phẩm thực hiện", "")
                    if sp:
                        st.markdown("**🧠 Sản phẩm / Nghiên cứu:**")
                        st.info(str(sp)[:300])

                st.markdown("</div>", unsafe_allow_html=True)
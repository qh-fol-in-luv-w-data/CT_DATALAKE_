import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ========================
# CONFIG
# ========================
MODEL_NAME = "intfloat/multilingual-e5-base"
SCORE_THRESHOLD = 0.83
model = SentenceTransformer(MODEL_NAME)

import os
BASE_PATH = os.path.dirname(__file__)

# ========================
# LOAD 2 MODE
# ========================
DATASETS = {
    "in": {
        "index": faiss.read_index(os.path.join(BASE_PATH, "index.faiss")),
        "metadata": json.load(open(os.path.join(BASE_PATH, "metadata.json"), "r", encoding="utf-8"))
    },
    "out": {
        "index": faiss.read_index(os.path.join(BASE_PATH, "index_out.faiss")),
        "metadata": json.load(open(os.path.join(BASE_PATH, "metadata_out.json"), "r", encoding="utf-8"))
    }
}

# ========================
# UTILS – MODE IN (tiếng Việt)
# ========================
def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return "" if x == 0 else str(x)
    return str(x).strip().lower()


def build_unique_key_in(item):
    """Ưu tiên ID → fallback sang (tên + trường + khoa)"""
    id_ = item.get("id")
    if id_:
        return f"id::{id_}"
    return "||".join([
        safe_str(item.get("tên")),
        safe_str(item.get("trường")),
        safe_str(item.get("khoa"))
    ])


def pass_filter_in(item, query_lower):
    hoc_ham = safe_str(item.get("học hàm"))
    hoc_vi  = safe_str(item.get("học vị"))
    chuc_vu = safe_str(item.get("chức vụ"))
    if "giáo sư"     in query_lower and "giáo sư"     not in hoc_ham: return False
    if "phó giáo sư" in query_lower and "phó giáo sư" not in hoc_ham: return False
    if "tiến sĩ"     in query_lower and "tiến sĩ"     not in hoc_vi:  return False
    if "thạc sĩ"     in query_lower and "thạc sĩ"     not in hoc_vi:  return False
    if "trưởng"      in query_lower and "trưởng"      not in chuc_vu: return False
    return True


# ========================
# UTILS – MODE OUT (Google Scholar / interests)
# ========================
def build_unique_key_out(item):
    """Dùng url làm key duy nhất, fallback sang name + affiliation"""
    url = item.get("url")
    if url:
        return f"url::{url}"
    return "||".join([
        safe_str(item.get("name")),
        safe_str(item.get("affiliation"))
    ])


def build_search_text_out(item):
    """
    Ghép interests + affiliation thành chuỗi để embed khi tạo index.
    Hàm này dùng để nhắc lại cách text đã được index – không gọi ở search time,
    chỉ để tham khảo / tạo lại index nếu cần.
    """
    interests  = ", ".join(item.get("interests") or [])
    affiliation = item.get("affiliation") or ""
    return f"{interests}. {affiliation}".strip(". ")


def pass_filter_out(item, query_lower):
    """
    Filter nhẹ dựa trên interests: nếu query chứa keyword rõ ràng
    thì ít nhất một interest phải chứa keyword đó (so khớp substring).
    Trả về True nếu không có keyword đặc thù hoặc có ít nhất một interest khớp.
    """
    interests_text = " ".join(item.get("interests") or []).lower()

    # Keyword mapping: query keyword → chuỗi cần có trong interests
    keyword_checks = {
        "machine learning"  : "machine learning",
        "deep learning"     : "deep learning",
        "artificial intelligence": "artificial intelligence",
        "ai"                : "ai",
        "nlp"               : "nlp",
        "natural language"  : "natural language",
        "computer vision"   : "computer vision",
        "data science"      : "data science",
        "finance"           : "financ",        # finance / financial
        "economics"         : "econom",        # economics / economic
        "management"        : "management",
        "tourism"           : "tourism",
        "chemistry"         : "chem",
        "physics"           : "physics",
        "biology"           : "biol",
        "civil engineering" : "civil",
        "software"          : "software",
        "network"           : "network",
        "security"          : "security",
    }

    for qkw, ikw in keyword_checks.items():
        if qkw in query_lower and ikw not in interests_text:
            return False
    return True


# ========================
# SEARCH
# ========================
def search(query: str, mode: str = "in", top_k: int = 5):
    """
    mode:
        in  -> index.faiss + metadata.json          (giảng viên nội bộ, tiếng Việt)
        out -> index_out.faiss + metadata_out.json  (Google Scholar, tìm theo interests)
    """
    if mode not in DATASETS:
        raise ValueError("mode phải là 'in' hoặc 'out'")

    index    = DATASETS[mode]["index"]
    metadata = DATASETS[mode]["metadata"]
    query_lower = query.lower()

    # ---------- encode query ----------
    if mode == "out":
        # Prefix "query:" dùng cho multilingual-e5 ở cả hai mode,
        # nhưng với out ta có thể thêm context "research interests:" để
        # định hướng embedding gần với cách interests được index.
        q_text = f"query: research interests related to {query}"
    else:
        q_text = f"query: {query}"

    q_vec = model.encode(
        [q_text],
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(q_vec, top_k * 5)

    # Global threshold: nếu kết quả tốt nhất không đủ điểm → trả rỗng
    if len(D[0]) == 0 or D[0][0] < SCORE_THRESHOLD:
        return []

    # ---------- chọn đúng helper theo mode ----------
    if mode == "out":
        build_key  = build_unique_key_out
        apply_filter = pass_filter_out
    else:
        build_key  = build_unique_key_in
        apply_filter = pass_filter_in

    results = []
    seen    = set()

    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        if score < SCORE_THRESHOLD:
            break

        item = metadata[idx]

        if not apply_filter(item, query_lower):
            continue

        key = build_key(item)
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "score": float(score),
            "data" : item
        })

        if len(results) >= top_k:
            break

    return results


# ========================
# DISPLAY HELPER – MODE OUT
# ========================
def display_out(item, score):
    interests = ", ".join(item.get("interests") or [])
    print(f"  Tên       : {item.get('name')}")
    print(f"  Affiliation: {item.get('affiliation')}")
    print(f"  Interests : {interests}")
    print(f"  Citations : {item.get('citations')}  |  h-index: {item.get('h_index')}")
    print(f"  URL       : {item.get('url')}")
    print(f"  Score     : {score:.4f}")


# ========================
# TEST
# ========================
if __name__ == "__main__":
    while True:
        mode = input("\nChọn mode (in/out): ").strip().lower()
        if mode not in ["in", "out"]:
            print("❌ Mode không hợp lệ")
            continue

        q = input("Nhập query: ").strip()
        if not q:
            break

        res = search(q, mode=mode, top_k=5)

        if not res:
            print("\n❌ Không tìm thấy kết quả phù hợp\n")
            continue

        print(f"\nKẾT QUẢ ({mode.upper()}):\n")
        for i, r in enumerate(res, 1):
            print(f"{i}. ", end="")
            if mode == "out":
                display_out(r["data"], r["score"])
            else:
                print(f"Score: {r['score']:.4f}")
                print(json.dumps(r["data"], ensure_ascii=False, indent=2))
            print("-" * 55)
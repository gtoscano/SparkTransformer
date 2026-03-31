import os, json, warnings, logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
import pandas as pd
import streamlit as st

# LangChain & OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS

st.set_page_config(
    page_title="CS Class Suggestion Advisor — Spring 2026",
    page_icon="🎓",
    layout="wide",
)


# ---------------------------
# Spring 2026 catalog (from your HTML)
# ---------------------------
def load_cua_spring_2026() -> pd.DataFrame:
    data = [
        # course_id, title, section, credits, level, days_times, room, instructor, dates, status
        [
            "CSC 104",
            "Introduction to Computers I",
            "01-LEC",
            3,
            100,
            "TBA (Online async)",
            "ONLINE ASYNCHRONOUS",
            "Hanney Shaban",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 107",
            "Introduction to Computer Security",
            "01-LEC",
            3,
            100,
            "TBA (Online async)",
            "ONLINE ASYNCHRONOUS",
            "Hanney Shaban",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 113",
            "Introduction to Computer Programming with MATLAB",
            "01-LEC",
            3,
            100,
            "MoWe 2:10PM–3:25PM",
            "Pangborn G023",
            "Gregory Behrmann",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 124",
            "Introduction to Computer Programming with Python",
            "01-LEC",
            3,
            100,
            "WeFr 10:40AM–11:55AM",
            "Pangborn 303",
            "Gregorio Toscano Pulido",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 223",
            "Object-Oriented Programming with Java",
            "01-LEC",
            3,
            200,
            "TuTh 11:10AM–12:25PM",
            "Pangborn 301",
            "Matthew Jacobs",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 306",
            "Introduction to Operating Systems",
            "01-LEC",
            3,
            300,
            "TuTh 2:10PM–3:25PM",
            "Pangborn 303",
            "Dominick Rizk",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 322",
            "Introduction to Computer Graphics",
            "01-LEC",
            3,
            300,
            "MoWe 2:10PM–3:25PM",
            "Pangborn G024",
            "Hieu Trung Bui",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 323",
            "Introduction to Computer Networks",
            "01-LEC",
            3,
            300,
            "TuTh 6:40PM–7:55PM",
            "Pangborn G035",
            "Sharif Khalil; Gahana Majumdar",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 326",
            "Switching Circuits and Logic Design",
            "01-LEC",
            3,
            300,
            "TuTh 12:40PM–1:55PM",
            "Gowan 401",
            "Dominick Rizk",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 327",
            "Switching Circuits and Logic Design Laboratory",
            "01-LAB",
            1,
            300,
            "TuTh 3:40PM–4:55PM",
            "Pangborn 207",
            "Thuc Phan",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 363",
            "Software Engineering",
            "01-LEC",
            3,
            300,
            "WeFr 11:10AM–12:25PM",
            "Pangborn G022",
            "Matthew Jacobs",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 406",
            "Introduction to Secure Computing",
            "01-LEC",
            3,
            400,
            "WeFr 3:40PM–4:55PM",
            "Pangborn 303",
            "Gregorio Toscano Pulido",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 409",
            "Web Design & Programming",
            "01-LEC",
            3,
            400,
            "We 5:10PM–7:40PM",
            "Pangborn 301",
            "Vladimir Kirnosov",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 411",
            "Analysis of Algorithms",
            "01-LEC",
            3,
            400,
            "TuTh 3:40PM–4:55PM",
            "Pangborn 303",
            "Minhee Jun; Andrew Heller",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 423",
            "Business Data Analytics",
            "01-LEC",
            3,
            400,
            "We 5:10PM–7:40PM",
            "ONLINE",
            "Aysegul Cuhadar",
            "01/13/2026–05/10/2026",
            "Closed",
        ],
        [
            "CSC 430",
            "Introduction to Data Analysis",
            "01-LEC",
            3,
            400,
            "We 5:10PM–7:40PM",
            "Pangborn G023",
            "Chaofan Sun",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 435",
            "Introduction to Deep Learning",
            "01-LAB",
            1,
            400,
            "Fr 5:10PM–7:40PM",
            "Pangborn 301",
            "Chaofan Sun",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 445",
            "Introduction to Data Mining",
            "01-LEC",
            3,
            400,
            "Th 5:10PM–7:40PM",
            "Pangborn G023",
            "Minhee Jun; Andrew Heller",
            "01/13/2026–05/10/2026",
            "Closed",
        ],
        [
            "CSC 484",
            "Introduction to Machine Learning",
            "01-LEC",
            3,
            400,
            "Mo 5:10PM–7:40PM",
            "Pangborn G023",
            "Chaofan Sun; Sairam Reddy Korimilla",
            "01/13/2026–05/10/2026",
            "Open",
        ],
        [
            "CSC 484",
            "Introduction to Machine Learning",
            "02-LEC",
            3,
            400,
            "Mo 12:40PM–3:10PM",
            "Pangborn 302",
            "Chaofan Sun; Sairam Reddy Korimilla",
            "01/13/2026–05/10/2026",
            "Open",
        ],
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "course_id",
            "title",
            "section",
            "credits",
            "level",
            "days_times",
            "room",
            "instructor",
            "dates",
            "status",
        ],
    )
    # Helpful derived fields
    df["when_offered"] = df["days_times"]
    df["prerequisites"] = ""  # not present in the HTML
    df["tags"] = ""
    df["description"] = df.apply(
        lambda r: f"{r['title']} — {r['days_times']} in {r['room']} (Instr: {r['instructor']}).",
        axis=1,
    )
    return df


def ensure_text(x):
    return "" if pd.isna(x) else str(x)


def build_vector_store(df: pd.DataFrame, embeddings: OpenAIEmbeddings) -> FAISS:
    texts, metas = [], []
    for _, row in df.iterrows():
        text = "\n".join(
            [
                f"{row['course_id']} {row['section']}: {row['title']}",
                f"Credits: {row['credits']} | Level: {row['level']} | Status: {row['status']}",
                f"When Offered: {ensure_text(row['when_offered'])} | Room: {ensure_text(row['room'])}",
                f"Instructor: {ensure_text(row['instructor'])}",
                f"Description: {ensure_text(row['description'])}",
            ]
        )
        texts.append(text)
        metas.append(
            {
                "course_id": row["course_id"],
                "title": row["title"],
                "section": row["section"],
                "credits": int(row["credits"]),
                "level": int(row["level"]),
                "status": row["status"],
            }
        )
    return FAISS.from_texts(texts, embedding=embeddings, metadatas=metas)


def parse_json_safely(text: str):
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return None


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("🔑 API & Data")
    api_key = st.text_input(
        "OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", "")
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.subheader("Catalog Source")
    data_mode = st.radio("Use:", ["Spring 2026 (from HTML)", "Upload CSV"], index=0)
    uploaded = None
    if data_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload catalog CSV", type=["csv"])
        st.caption(
            "Columns expected: course_id,title,section,credits,level,when_offered,room,instructor,dates,status,description"
        )

    hide_closed = st.checkbox("Hide CLOSED sections", value=True)

# ---------------------------
# Load catalog
# ---------------------------
if data_mode == "Upload CSV" and uploaded is not None:
    catalog_df = pd.read_csv(uploaded)
else:
    catalog_df = load_cua_spring_2026()

if hide_closed and "status" in catalog_df.columns:
    catalog_df = catalog_df[catalog_df["status"].str.lower() != "closed"].copy()

# Coerce types
for c in ("credits", "level"):
    if c in catalog_df.columns:
        catalog_df[c] = (
            pd.to_numeric(catalog_df[c], errors="coerce").fillna(3).astype(int)
        )

st.title("🎓 CS Class Suggestion Advisor (Spring 2026)")
st.caption(
    "Semantic search over your Spring 2026 CS classes + an LLM that returns structured, explainable picks."
)

st.subheader("Catalog Preview")
st.dataframe(
    catalog_df[
        [
            "course_id",
            "title",
            "section",
            "when_offered",
            "room",
            "instructor",
            "status",
        ]
    ],
    use_container_width=True,
)

# ---------------------------
# Build index
# ---------------------------
if "faiss" not in st.session_state:
    st.session_state.faiss = None
embeddings_model = "text-embedding-3-large"
chat_model = "gpt-4o-mini"

col1, col2 = st.columns(2)
with col1:
    if st.button("🔧 Build / Rebuild Index"):
        if not api_key:
            st.error("Add your OPENAI_API_KEY in the sidebar.")
        else:
            with st.spinner("Building FAISS index…"):
                st.session_state.faiss = build_vector_store(
                    catalog_df, OpenAIEmbeddings(model=embeddings_model)
                )
            st.success("Index built ✅")
with col2:
    st.write(
        "Index status:",
        "✅ Ready" if st.session_state.faiss is not None else "⚠️ Not built",
    )

st.divider()

# ---------------------------
# Inputs
# ---------------------------
st.subheader("Student Profile & Constraints")
interests = st.text_area(
    "Interests (topics, careers, tools)",
    placeholder="e.g., deep learning, security, networks, web dev, operating systems",
)

completed = st.text_area(
    "Completed Courses (IDs or names)",
    placeholder="e.g., CSC 124, Calculus I, Data Structures",
)

c1, c2, c3 = st.columns(3)
with c1:
    target_credits = st.number_input(
        "Target total credits", min_value=3, max_value=21, value=6, step=1
    )
with c2:
    min_level = st.selectbox("Min level", [100, 200, 300, 400], index=0)
with c3:
    max_level = st.selectbox("Max level", [200, 300, 400], index=2)

schedule_blockers = st.text_input(
    "Schedule blockers (free text)",
    placeholder="e.g., prefer evenings; no Tu/Th before 3pm; avoid Fri night",
)

n_recs = st.slider("Number of recommendations", 1, 8, 5)


# ---------------------------
# Retrieval + LLM
# ---------------------------
def retrieve_candidates(query: str, k: int = 12):
    if st.session_state.faiss is None:
        return []
    return st.session_state.faiss.similarity_search(query, k=k)


def filter_for_bounds(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    m = (df["level"] >= lo) & (df["level"] <= hi)
    return df[m].copy()


def candidate_table(df: pd.DataFrame, q: str, k: int = 16):
    if st.session_state.faiss is not None and q.strip():
        docs = retrieve_candidates(q, k=k)
        order = [
            d.metadata.get("course_id") + ":" + d.metadata.get("section") for d in docs
        ]
        key = df["course_id"] + ":" + df["section"]
        sub = df[key.isin(order)].copy()
        sub["__rank"] = key[key.isin(order)].apply(lambda s: order.index(s))
        return sub.sort_values("__rank").drop(columns="__rank")
    if not q.strip():
        return df.copy()
    mask = df["description"].str.contains(
        "|".join([w.strip() for w in q.split(",") if w.strip()]), case=False, na=False
    )
    return df[mask].copy() if mask.any() else df.copy()


def llm_recommend(rows: list, student_profile: dict, n: int = 5):
    system = SystemMessage(
        content=(
            "You are a meticulous CS academic advisor for Spring 2026. "
            "Respect credits, level bounds, schedule blockers, section status (prefer Open), and common prerequisites. "
            "Return STRICT JSON with fields: summary, total_credits, recommendations[]. "
            "Each recommendation must include course_id, title, section, credits, fit_reasoning, prereq_status, schedule_fit, workload_note."
        )
    )

    schema_hint = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "total_credits": {"type": "number"},
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "course_id": {"type": "string"},
                        "title": {"type": "string"},
                        "section": {"type": "string"},
                        "credits": {"type": "number"},
                        "fit_reasoning": {"type": "string"},
                        "prereq_status": {"type": "string"},
                        "schedule_fit": {"type": "string"},
                        "workload_note": {"type": "string"},
                    },
                    "required": [
                        "course_id",
                        "title",
                        "section",
                        "credits",
                        "fit_reasoning",
                        "prereq_status",
                        "schedule_fit",
                        "workload_note",
                    ],
                },
            },
        },
        "required": ["summary", "total_credits", "recommendations"],
    }

    compact_catalog = [
        {
            "course_id": r.get("course_id"),
            "title": r.get("title"),
            "section": r.get("section"),
            "credits": r.get("credits"),
            "level": r.get("level"),
            "status": r.get("status", "Open"),
            "when_offered": r.get("when_offered", ""),
            "room": r.get("room", ""),
            "instructor": r.get("instructor", ""),
            "description": (r.get("description", "") or "")[:600],
        }
        for r in rows
    ]

    user = HumanMessage(
        content=json.dumps(
            {
                "schema": schema_hint,
                "term": "Spring 2026",
                "n_recommendations": n,
                "student_profile": student_profile,
                "candidate_courses": compact_catalog,
            }
        )
    )

    llm = ChatOpenAI(model=chat_model, temperature=0.2)
    resp = llm.invoke([system, user])
    return parse_json_safely(resp.content)


if st.button("🎯 Suggest Classes"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set your OPENAI_API_KEY in the sidebar.")
    else:
        bounded = filter_for_bounds(catalog_df, min_level, max_level)
        cands = candidate_table(bounded, interests, k=20)
        rows = cands.to_dict(orient="records")
        profile = {
            "interests": interests,
            "completed_courses": completed,
            "target_credits": target_credits,
            "level_bounds": [min_level, max_level],
            "schedule_blockers": schedule_blockers,
            "prefer_open_sections": True,
        }
        with st.spinner("Generating recommendations…"):
            result = llm_recommend(rows, profile, n=n_recs)
        if not result:
            st.error(
                "Could not parse model output. Try fewer constraints or rebuild the index."
            )
        else:
            st.subheader("Recommendations")
            st.write(result.get("summary", ""))
            recs = result.get("recommendations", [])
            total = result.get("total_credits", 0)
            if recs:
                out = pd.DataFrame(recs)[
                    [
                        "course_id",
                        "title",
                        "section",
                        "credits",
                        "fit_reasoning",
                        "prereq_status",
                        "schedule_fit",
                        "workload_note",
                    ]
                ]
                st.dataframe(out, use_container_width=True, height=360)
                st.info(f"Proposed total credits: **{total}**")
            else:
                st.warning(
                    "No courses matched all constraints. Consider broadening level range or credits."
                )

st.divider()
st.markdown("""**How it’s tailored for Spring 2026**
- Built-in catalog and meeting details from your file (status-aware; CLOSED hidden by default).
- Semantic retrieval (FAISS + OpenAI embeddings) over course descriptions and meeting notes.
- LLM returns strict JSON with reasoning, prereq hints, and schedule fit.
""")

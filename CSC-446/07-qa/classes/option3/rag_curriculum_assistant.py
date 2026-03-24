import os
import json
from typing import Any, Dict, List
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# ---------- Load JSON curriculum ----------
with open("../cua_bscs_curriculum.json", "r", encoding="utf-8") as f:
    DATA = json.load(f)

PROGRAMS = DATA["programs"]
SEMESTERS = DATA["semesters"]
COURSES = DATA["courses"]
SEMESTER_COURSES = DATA["semester_courses"]

# Lookup maps
COURSE_BY_CODE = {c["code"]: c for c in COURSES}
SEMESTER_BY_ID = {s["id"]: s for s in SEMESTERS}

# ---------- Curriculum tools (structured JSON) ----------

def get_programs() -> List[Dict[str, Any]]:
    return PROGRAMS

def get_semesters(program_id: str) -> List[Dict[str, Any]]:
    return [s for s in SEMESTERS if s["program_id"] == program_id]

def get_semester(program_id: str, year: int, term: str) -> Dict[str, Any]:
    term = term.capitalize()
    sem = None
    for s in SEMESTERS:
        if (s["program_id"] == program_id and
            s["year"] == year and
            s["term"].lower() == term.lower()):
            sem = s
            break
    if sem is None:
        return {"error": "Semester not found."}

    sem_id = sem["id"]
    courses_in_sem = []
    for sc in SEMESTER_COURSES:
        if sc["semester_id"] == sem_id:
            c = COURSE_BY_CODE.get(sc["course_code"])
            if c:
                courses_in_sem.append({
                    "code": c["code"],
                    "title": c["title"],
                    "credits": sc["credits"],
                })
    return {"semester": sem, "courses": courses_in_sem}

def get_course(course_code: str) -> Dict[str, Any]:
    course_code_lower = course_code.strip().lower()
    for c in COURSES:
        if c["code"].lower() == course_code_lower:
            return c
    return {"error": f"Course {course_code} not found."}

def list_semesters_for_course(course_code: str) -> List[Dict[str, Any]]:
    course_code_lower = course_code.strip().lower()
    sem_ids = set(
        sc["semester_id"] for sc in SEMESTER_COURSES
        if sc["course_code"].lower() == course_code_lower
    )
    result = []
    for s in SEMESTERS:
        if s["id"] in sem_ids:
            result.append({
                "semester_id": s["id"],
                "program_id": s["program_id"],
                "year": s["year"],
                "term": s["term"],
                "total_credits": s["total_credits"],
            })
    return result

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_programs",
            "description": "List all BSCS program models (e.g., 2017–2023 and 2023+).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_semesters",
            "description": "List all semesters for a given program_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_id": {"type": "string"},
                },
                "required": ["program_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_semester",
            "description": "Get a specific semester and its courses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_id": {"type": "string"},
                    "year": {"type": "integer"},
                    "term": {
                        "type": "string",
                        "enum": ["Fall", "Spring"],
                    },
                },
                "required": ["program_id", "year", "term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_course",
            "description": "Get details for a course by course_code, e.g., 'CSC 123'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_code": {"type": "string"},
                },
                "required": ["course_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_semesters_for_course",
            "description": "List semesters (program/year/term) where a course appears.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_code": {"type": "string"},
                },
                "required": ["course_code"],
            },
        },
    },
]

TOOL_IMPLS = {
    "get_programs": lambda args: get_programs(),
    "get_semesters": lambda args: get_semesters(args["program_id"]),
    "get_semester": lambda args: get_semester(args["program_id"], args["year"], args["term"]),
    "get_course": lambda args: get_course(args["course_code"]),
    "list_semesters_for_course": lambda args: list_semesters_for_course(args["course_code"]),
}

# ---------- RAG (Chroma) setup ----------
DB_PATH = "rag_db"
COLLECTION_NAME = "cs_syllabi_advising"
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI()
chroma_client = chromadb.PersistentClient(path=DB_PATH, settings=Settings())
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# ---------- System prompt ----------
SYSTEM_PROMPT = """
You are an academic advisor for the Bachelor of Science in Computer Science
program at The Catholic University of America (CUA).

You have TWO kinds of knowledge:

1) Structured JSON curriculum data available via tools:
   - Program versions (2017–2023, 2023+)
   - Semesters (year, term, total credits)
   - Courses (code, title, credits)
   Use these tools for any questions requiring factual details like credits,
   course lists, or which semester a course appears in.

2) Retrieved text passages (“Context documents”) from syllabi and advising rules:
   - Course descriptions, topics, learning outcomes, and advising guidelines.

RULES:
- For credits, course lists per semester, and required sequence: rely on TOOLS.
- For what a course is about, typical workload, or advising guidance:
  rely on the provided Context documents and your reasoning.
- If tools and context ever differ on curriculum facts, TOOLS are the source of truth.
- If the user does not specify which program version they are under (2017–2023 vs 2023+),
  politely ask them to clarify.
- If something is not contained in tools or context documents, say that information is not available.
- Answer clearly, with bullet points or short paragraphs, and explain your reasoning when helpful.
"""

# ---------- Helper: RAG retrieval ----------
def retrieve_context(query: str, k: int = 4) -> str:
    emb_resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
    )
    q_emb = emb_resp.data[0].embedding

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    parts = []
    for doc, md in zip(docs, metas):
        src = md.get("source_type", "unknown")
        path = md.get("rel_path", "")
        parts.append(f"[Source: {src} | {path}]\n{doc.strip()}\n")

    if not parts:
        return "No relevant context documents were retrieved."
    return "\n\n".join(parts)

# ---------- Main Q&A ----------
def ask_llm(user_question: str) -> str:
    # 1) Retrieve RAG context
    context_text = retrieve_context(user_question, k=4)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "Context documents (from syllabi and advising rules):\n\n" + context_text,
        },
        {"role": "user", "content": user_question},
    ]

    # 2) First call: model can decide to use tools
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = resp.choices[0].message

    if not msg.tool_calls:
        return msg.content

    # 3) Handle tool calls
    tool_messages = []
    for tc in msg.tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")
        impl = TOOL_IMPLS.get(name)
        if impl is None:
            result = {"error": f"Unknown tool {name}"}
        else:
            result = impl(args)

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(result),
            }
        )

    messages.append(msg)
    messages.extend(tool_messages)

    # 4) Second call: final answer
    resp2 = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
    )
    return resp2.choices[0].message.content

# ---------- CLI ----------
if __name__ == "__main__":
    print("Hybrid RAG + JSON curriculum assistant. Ctrl+C to exit.")
    while True:
        try:
            q = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        ans = ask_llm(q)
        print("\nAssistant:\n", ans)


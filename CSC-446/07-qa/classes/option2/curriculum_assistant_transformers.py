import os
import re
import json
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -------------------------
# 0. Config: choose your local model
# -------------------------

MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")

use_cuda = torch.cuda.is_available()
print(f"Using model: {MODEL_NAME}")
print(f"Device: {'cuda' if use_cuda else 'cpu'}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dtype = torch.float16 if use_cuda else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto" if use_cuda else None,
)

# IMPORTANT: no `device=` here because accelerate is managing devices
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=False,  # deterministic
)


def call_model(prompt: str, max_new_tokens: int = 256) -> str:
    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"]
    return out[len(prompt):].strip()


# -------------------------
# 1. Load JSON dataset
# -------------------------

with open("../cua_bscs_curriculum.json", "r", encoding="utf-8") as f:
    DATA = json.load(f)

PROGRAMS = DATA["programs"]
SEMESTERS = DATA["semesters"]
COURSES = DATA["courses"]
SEMESTER_COURSES = DATA["semester_courses"]


# -------------------------
# 2. Local Python “tools”
# -------------------------

def get_programs() -> List[Dict[str, Any]]:
    return PROGRAMS


def get_semesters(program_id: str) -> List[Dict[str, Any]]:
    return [s for s in SEMESTERS if s["program_id"] == program_id]


def get_semester(program_id: str, year: int, term: str) -> Dict[str, Any]:
    term = term.capitalize()
    sem = None
    for s in SEMESTERS:
        if (
            s["program_id"] == program_id
            and s["year"] == year
            and s["term"].lower() == term.lower()
        ):
            sem = s
            break

    if sem is None:
        return {"error": "Semester not found."}

    sem_id = sem["id"]

    courses_in_sem = []
    for sc in SEMESTER_COURSES:
        if sc["semester_id"] == sem_id:
            c = next((c for c in COURSES if c["code"] == sc["course_code"]), None)
            if c:
                courses_in_sem.append(
                    {
                        "code": c["code"],
                        "title": c["title"],
                        "credits": sc["credits"],
                    }
                )

    return {
        "semester": sem,
        "courses": courses_in_sem,
    }


def get_course(course_code: str) -> Dict[str, Any]:
    course_code_lower = course_code.lower().strip()
    for c in COURSES:
        if c["code"].lower() == course_code_lower:
            return c
    return {"error": f"Course {course_code} not found."}


def list_semesters_for_course(course_code: str) -> List[Dict[str, Any]]:
    course_code_lower = course_code.lower().strip()
    sem_ids = set()
    for sc in SEMESTER_COURSES:
        if sc["course_code"].lower() == course_code_lower:
            sem_ids.add(sc["semester_id"])

    result = []
    for s in SEMESTERS:
        if s["id"] in sem_ids:
            result.append(
                {
                    "semester_id": s["id"],
                    "program_id": s["program_id"],
                    "year": s["year"],
                    "term": s["term"],
                    "total_credits": s["total_credits"],
                }
            )
    return result


# -------------------------
# 3. Simple router in Python (no JSON tool protocol)
# -------------------------

PROGRAM_ID_2017_2023 = "bscs-2017-2023"
PROGRAM_ID_2023_PLUS = "bscs-2023+"

def infer_program_id(question: str) -> Optional[str]:
    q = question.lower()
    if "2017" in q or "2018" in q or "2019" in q or "2020" in q or "2021" in q or "2022" in q or "2023" in q and "entering in fall 2017" in q:
        return PROGRAM_ID_2017_2023
    if "2023+" in q or "2023 +" in q or "entering in fall 2023" in q or "new curriculum" in q:
        return PROGRAM_ID_2023_PLUS
    # If the question explicitly says "2017–2023 curriculum"
    if "2017-2023" in q or "2017–2023" in q:
        return PROGRAM_ID_2017_2023
    # If the question explicitly says "2023+ curriculum"
    if "2023 curriculum" in q:
        return PROGRAM_ID_2023_PLUS
    return None


def parse_year_and_term(question: str) -> Optional[Dict[str, Any]]:
    # Look for pattern like "Year 2 Spring", "year 3 fall"
    m = re.search(r"year\s+(\d+)\s+(fall|spring)", question, re.IGNORECASE)
    if not m:
        return None
    year = int(m.group(1))
    term = m.group(2).capitalize()
    return {"year": year, "term": term}


def detect_course_code(question: str) -> Optional[str]:
    # crude: look for patterns like CSC 210, MATH 121, etc.
    m = re.search(r"\b([A-Z]{2,4}\s*\d{3})\b", question)
    if not m:
        return None
    code = m.group(1).upper().replace("  ", " ").strip()
    return code


# -------------------------
# 4. Answer generators (LLM used only for phrasing)
# -------------------------

def answer_from_semester(question: str, sem_data: Dict[str, Any]) -> str:
    if "error" in sem_data:
        return sem_data["error"]

    sem = sem_data["semester"]
    courses = sem_data["courses"]
    total_credits = sem.get("total_credits")

    # Build a small context for the model
    context_lines = []
    context_lines.append(
        f"Program: {sem['program_id']}, Year {sem['year']} {sem['term']}"
    )
    context_lines.append("Courses in this semester:")
    credit_sum = 0
    for c in courses:
        context_lines.append(
            f"- {c['code']}: {c['title']} — {c['credits']} credits"
        )
        credit_sum += c["credits"]

    context_lines.append(f"Total credits (from curriculum JSON): {total_credits}")
    context_lines.append(f"Total credits (sum of listed courses): {credit_sum}")

    context = "\n".join(context_lines)

    prompt = f"""You are an academic advising assistant for the CS program.

You are given curriculum data for a specific semester:

{context}

User question:
{question}

Using ONLY the data above, answer the user's question. 
List the courses with their credits and clearly state the total number of credits.
Answer concisely.

Answer:
"""

    return call_model(prompt, max_new_tokens=256)


def answer_from_course(question: str, course_data: Dict[str, Any], sem_list: List[Dict[str, Any]]) -> str:
    if "error" in course_data:
        return course_data["error"]

    ctx_lines = []
    ctx_lines.append(f"Course code: {course_data['code']}")
    ctx_lines.append(f"Title: {course_data.get('title', '(no title)')}")
    ctx_lines.append(f"Default credits: {course_data.get('credits_default')}")
    ctx_lines.append("Semesters where this course appears:")

    if not sem_list:
        ctx_lines.append("- (no semesters found in curriculum JSON)")
    else:
        for s in sem_list:
            ctx_lines.append(
                f"- Program: {s['program_id']}, Year {s['year']} {s['term']} "
                f"(total semester credits: {s['total_credits']})"
            )

    context = "\n".join(ctx_lines)

    prompt = f"""You are an academic advising assistant for the CS program.

You are given curriculum data about a single course and the semesters where it appears:

{context}

User question:
{question}

Using ONLY the data above, answer the user's question clearly and concisely.
If the user asks about prerequisites or topics and they are not present, say so.

Answer:
"""

    return call_model(prompt, max_new_tokens=256)


def answer_generic(question: str) -> str:
    prompt = f"""You are an academic advising assistant for the Bachelor of Science in Computer Science
at The Catholic University of America.

The user asked:

{question}

You do not have any structured curriculum data for this question, but answer it in a general,
helpful way, making reasonable assumptions for a typical 4-year CS program (without giving
specific course codes or credits).

Answer:
"""
    return call_model(prompt, max_new_tokens=256)


# -------------------------
# 5. Main router: Python decides what to do
# -------------------------

def ask_llm(user_question: str) -> str:
    q = user_question.strip()
    if not q:
        return "Please enter a question."

    # 1) Try to parse "Year X Term" → get_semester
    yt = parse_year_and_term(q)
    program_id = infer_program_id(q)

    if yt and program_id:
        sem_data = get_semester(program_id, yt["year"], yt["term"])
        return answer_from_semester(q, sem_data)

    # 2) Try to parse a course code → course + its semesters
    course_code = detect_course_code(q)
    if course_code:
        course_data = get_course(course_code)
        sem_list = list_semesters_for_course(course_code)
        return answer_from_course(q, course_data, sem_list)

    # 3) Fallback generic answer (no tools)
    return answer_generic(q)


# -------------------------
# 6. Simple CLI
# -------------------------

if __name__ == "__main__":
    print("Curriculum assistant (transformers + local tools, routed by Python). Ctrl+C to exit.")
    while True:
        try:
            q = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        answer = ask_llm(q)
        print("\nAssistant:\n", answer)


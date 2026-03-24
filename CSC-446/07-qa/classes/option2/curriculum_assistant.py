import os
import json
from typing import List, Dict, Any
from openai import OpenAI

# -------------------------
# 0. Load JSON dataset
# -------------------------

with open("../cua_bscs_curriculum.json", "r", encoding="utf-8") as f:
    DATA = json.load(f)

PROGRAMS = DATA["programs"]
SEMESTERS = DATA["semesters"]
COURSES = DATA["courses"]
SEMESTER_COURSES = DATA["semester_courses"]

# -------------------------
# 1. Local Python functions
# -------------------------

def get_programs() -> List[Dict[str, Any]]:
    """Return list of all BSCS program models."""
    return PROGRAMS


def get_semesters(program_id: str) -> List[Dict[str, Any]]:
    """Return all semesters for a specific program_id."""
    return [
        s for s in SEMESTERS
        if s["program_id"] == program_id
    ]


def get_semester(program_id: str, year: int, term: str) -> Dict[str, Any]:
    """
    Return one semester with its courses.
    term: "Fall" or "Spring"
    """
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

    # get all semester-course links
    courses_in_sem = []
    for sc in SEMESTER_COURSES:
        if sc["semester_id"] == sem_id:
            # find course details
            c = next((c for c in COURSES if c["code"] == sc["course_code"]), None)
            if c:
                courses_in_sem.append({
                    "code": c["code"],
                    "title": c["title"],
                    "credits": sc["credits"]
                })

    return {
        "semester": sem,
        "courses": courses_in_sem
    }


def get_course(course_code: str) -> Dict[str, Any]:
    """Return details of a specific course by code."""
    course_code_lower = course_code.lower().strip()
    for c in COURSES:
        if c["code"].lower() == course_code_lower:
            return c
    return {"error": f"Course {course_code} not found."}


def list_semesters_for_course(course_code: str) -> List[Dict[str, Any]]:
    """
    Return all semesters where the given course appears.
    Each item includes program_id, year, term, and semester_id.
    """
    course_code_lower = course_code.lower().strip()

    # find all semester_ids where this course appears
    sem_ids = set()
    for sc in SEMESTER_COURSES:
        if sc["course_code"].lower() == course_code_lower:
            sem_ids.add(sc["semester_id"])

    # map back to semester objects
    result = []
    for s in SEMESTERS:
        if s["id"] in sem_ids:
            result.append({
                "semester_id": s["id"],
                "program_id": s["program_id"],
                "year": s["year"],
                "term": s["term"],
                "total_credits": s["total_credits"]
            })

    return result


# -------------------------
# 2. Tool definitions for the LLM
# -------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_programs",
            "description": "List all BSCS program models (e.g., 2017–2023 and 2023+).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_semesters",
            "description": "List all semesters for a given program_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_id": {
                        "type": "string",
                        "description": "Program ID, e.g., 'bscs-2017-2023' or 'bscs-2023+'."
                    }
                },
                "required": ["program_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_semester",
            "description": "Get a specific semester and its courses for a given program, year, and term.",
            "parameters": {
                "type": "object",
                "properties": {
                    "program_id": {
                        "type": "string",
                        "description": "Program ID, e.g., 'bscs-2017-2023' or 'bscs-2023+'."
                    },
                    "year": {
                        "type": "integer",
                        "description": "Year in the curriculum (1 to 4)."
                    },
                    "term": {
                        "type": "string",
                        "enum": ["Fall", "Spring"],
                        "description": "Semester term."
                    }
                },
                "required": ["program_id", "year", "term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_course",
            "description": "Get details for a course by course_code, e.g., 'CSC 123'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_code": {
                        "type": "string"
                    }
                },
                "required": ["course_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_semesters_for_course",
            "description": "List all semesters (program/year/term) where a course appears.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_code": {
                        "type": "string"
                    }
                },
                "required": ["course_code"]
            }
        }
    }
]

# Map tool names to Python functions
TOOL_IMPLS = {
    "get_programs": lambda args: get_programs(),
    "get_semesters": lambda args: get_semesters(args["program_id"]),
    "get_semester": lambda args: get_semester(args["program_id"], args["year"], args["term"]),
    "get_course": lambda args: get_course(args["course_code"]),
    "list_semesters_for_course": lambda args: list_semesters_for_course(args["course_code"]),
}

# -------------------------
# 3. OpenAI client
# -------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an academic advising assistant for the Bachelor of Science in Computer Science
program at The Catholic University of America.

You have access to tools that return authoritative data from the official curriculum JSON.
Use these tools to answer questions about:

- Programs (2017–2023, 2023+)
- Semesters (courses, credits, total credits)
- Courses (titles, credits, which semesters they appear in)

Rules:
1. Always call a tool when specific factual curriculum data is needed.
2. If the user does not specify the program version, ask for clarification.
3. Never invent courses or credits. Only use tool outputs.
4. Present answers clearly, using bullet lists or tables when helpful.
"""

# -------------------------
# 4. Helper: one round of Q&A
# -------------------------

def ask_llm(user_question: str):
    """
    Single-turn interaction:
    - Send user question + tools
    - If the model calls tools, run them
    - Send the tool outputs back
    - Return the final assistant message
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    # First call: allow the model to decide whether to call tools
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = resp.choices[0].message

    # If no tool calls, just return the reply
    if not msg.tool_calls:
        return msg.content

    # Otherwise, handle tool calls
    tool_messages = []
    for tool_call in msg.tool_calls:
        tool_name = tool_call.function.name
        args_json = tool_call.function.arguments
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            args = {}

        impl = TOOL_IMPLS.get(tool_name)
        if impl is None:
            tool_result = {"error": f"Unknown tool {tool_name}"}
        else:
            tool_result = impl(args)

        # Append tool result as a message
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result),
            }
        )

    # Second call: give the model the tool outputs so it can write the final answer
    messages.append(msg)  # model's tool_call message
    messages.extend(tool_messages)

    resp2 = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
    )

    final_msg = resp2.choices[0].message
    return final_msg.content


# -------------------------
# 5. Simple CLI
# -------------------------

if __name__ == "__main__":
    print("Curriculum assistant. Ask a question (Ctrl+C to exit).")
    while True:
        try:
            q = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        answer = ask_llm(q)
        print("\nAssistant:\n", answer)


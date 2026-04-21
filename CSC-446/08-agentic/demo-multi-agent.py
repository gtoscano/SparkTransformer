"""
Multi-agent demo: a router agent delegates to two specialist agents.

- course_agent: handles course lookups and professor office hours
- grade_events_agent: handles grade calculations and campus events
- router: decides which specialist(s) to call

This uses a manual routing approach that works reliably with small models.
The router is itself an LLM that picks the right specialist(s).

Run:
  python demo-multi-agent.py
"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype="auto",
)

# ── Mock data ──────────────────────────────────────────

COURSES = {
    "CSC101": {"title": "Intro to Computer Science", "schedule": "MWF 9:00–9:50 AM", "room": "Pangborn 107"},
    "CSC202": {"title": "Data Structures", "schedule": "TR 11:00–12:15 PM", "room": "Pangborn 220"},
    "CSC446": {"title": "Natural Language Processing", "schedule": "TR 2:00–3:15 PM", "room": "Pangborn 312"},
    "LLM500": {"title": "Large Language Models Seminar", "schedule": "W 4:00–6:30 PM", "room": "Hannan 105"},
}

PROFESSORS = {
    "toscano": {"name": "Dr. Gregorio Toscano", "hours": "TR 3:30–5:00 PM", "location": "Pangborn 301A"},
    "lin-ching": {"name": "Dr. Lin-Ching Chang", "hours": "MWF 10:00–11:00 AM", "location": "Pangborn 205"},
    "sierra": {"name": "Dr. Sierra", "hours": "MW 1:00–2:30 PM", "location": "Pangborn 310"},
}

EVENTS = {
    "2026-04-18": ["Spring Research Symposium — Great Hall, 10 AM–3 PM", "Career Fair — Pryzbyla Center, 4–7 PM"],
    "2026-04-19": ["CUA Hackathon — Pangborn Lab, 9 AM–9 PM"],
    "2026-04-21": ["Guest Lecture: AI in Healthcare — Hannan 105, 5 PM"],
    "2026-04-25": ["Last day of classes"],
}

# ── Course specialist tools ────────────────────────────

def lookup_course(course_id: str) -> str:
    """Look up a course by its ID and return the title, schedule, and room.

    Args:
        course_id: The course identifier to look up (e.g. 'CSC446').
    """
    info = COURSES.get(course_id.upper().replace(" ", ""))
    if not info:
        return f"No course found for '{course_id}'. Available: {', '.join(COURSES.keys())}."
    return f"{course_id.upper()} — {info['title']}\nSchedule: {info['schedule']}\nRoom: {info['room']}"


def office_hours(professor: str) -> str:
    """Return office hours and location for a professor. Use last name in lowercase.

    Args:
        professor: The professor's last name in lowercase (e.g. 'toscano').
    """
    key = professor.strip().lower()
    info = PROFESSORS.get(key)
    if not info:
        return f"No record for '{professor}'. Available: {', '.join(PROFESSORS.keys())}."
    return f"{info['name']}\nOffice hours: {info['hours']}\nLocation: {info['location']}"


# ── Grade & events specialist tools ────────────────────

def grade_calculator(scores: str, weights: str) -> str:
    """Compute a weighted average. Scores and weights are comma-separated numbers.

    Args:
        scores: Comma-separated scores (e.g. '85,92').
        weights: Comma-separated weights that sum to 1.0 (e.g. '0.4,0.6').
    """
    try:
        s = [float(x.strip()) for x in scores.split(",")]
        w = [float(x.strip()) for x in weights.split(",")]
        if len(s) != len(w):
            return f"Error: got {len(s)} scores but {len(w)} weights."
        if abs(sum(w) - 1.0) > 0.01:
            return f"Warning: weights sum to {sum(w):.2f}, not 1.0. Computing anyway."
        result = sum(si * wi for si, wi in zip(s, w))
        return f"Weighted average: {result:.2f}"
    except Exception as e:
        return f"Error: {e}"


def campus_events(date: str) -> str:
    """Return campus events for a given date (YYYY-MM-DD format).

    Args:
        date: The date to check in YYYY-MM-DD format (e.g. '2026-04-18').
    """
    events = EVENTS.get(date)
    if not events:
        return f"No events found for {date}."
    return f"Events on {date}:\n" + "\n".join(f"  - {e}" for e in events)


# ── Specialist agents ──────────────────────────────────
# Each specialist is an agent loop with its own tools and system prompt.

COURSE_TOOLS = {"lookup_course": lookup_course, "office_hours": office_hours}
GRADE_TOOLS = {"grade_calculator": grade_calculator, "campus_events": campus_events}

def run_specialist(system_prompt, tools_dict, tool_functions, user_text):
    """Run a specialist agent loop with its own tools."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    for _ in range(5):
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tool_functions,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        tool_calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)

        if not tool_calls:
            return response

        messages.append({"role": "assistant", "content": response})


        for tc_json in tool_calls:
            try:
                tool_request = json.loads(tc_json)
            except json.JSONDecodeError:
                continue

            tool_name = tool_request.get("name") or tool_request.get("tool_name", "")
            arguments = tool_request.get("arguments", {})

            if tool_name not in tools_dict:
                tool_result = f"Error: unknown tool '{tool_name}'."
            else:
                tool_result = tools_dict[tool_name](**arguments)

            messages.append({"role": "tool", "name": tool_name, "content": tool_result})

    return response


def course_agent(question):
    """Specialist for courses and professor office hours."""
    print("  [router] → course_agent")
    return run_specialist(
        "You are a course information specialist. Answer questions about courses and professor office hours. Be concise.",
        COURSE_TOOLS,
        [lookup_course, office_hours],
        question,
    )


def grade_events_agent(question):
    """Specialist for grade calculations and campus events."""
    print("  [router] → grade_events_agent")
    return run_specialist(
        "You are a grade and events specialist. Compute grades and look up campus events. Be concise.",
        GRADE_TOOLS,
        [grade_calculator, campus_events],
        question,
    )


# ── Router ─────────────────────────────────────────────
# The router is an LLM call that classifies the question
# and picks which specialist(s) to invoke.

ROUTER_PROMPT = """\
You are a router. Given a user question, decide which specialist(s) should handle it.

Available specialists:
- course_agent: course schedules, rooms, professor office hours
- grade_events_agent: grade calculations, campus events

Reply with ONLY a JSON list of specialist names. Examples:
- ["course_agent"]
- ["grade_events_agent"]
- ["course_agent", "grade_events_agent"]

User question: {question}
"""


def route(question):
    """Use the LLM to classify which specialist(s) to call."""
    messages = [{"role": "user", "content": ROUTER_PROMPT.format(question=question)}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Parse the JSON list from the response
    try:
        # Try to find a JSON array in the response
        match = re.search(r'\[.*?\]', response)
        if match:
            agents = json.loads(match.group())
            return [a for a in agents if a in ("course_agent", "grade_events_agent")]
    except json.JSONDecodeError:
        pass

    # Fallback: keyword matching
    agents = []
    q_lower = question.lower()
    if any(kw in q_lower for kw in ("course", "class", "schedule", "professor", "office hour", "room", "csc", "llm")):
        agents.append("course_agent")
    if any(kw in q_lower for kw in ("grade", "score", "midterm", "final", "weight", "event", "happening", "campus")):
        agents.append("grade_events_agent")
    return agents or ["course_agent"]


SPECIALISTS = {
    "course_agent": course_agent,
    "grade_events_agent": grade_events_agent,
}


# ── Manager: route + collect + synthesize ──────────────

def run_manager(question):
    """Route the question, call specialists, combine answers."""
    agents_to_call = route(question)
    print(f"  [router] Selected: {agents_to_call}")

    results = []
    for agent_name in agents_to_call:
        answer = SPECIALISTS[agent_name](question)
        results.append(answer)

    if len(results) == 1:
        return results[0]

    # If multiple specialists answered, ask the LLM to combine
    combined_context = "\n\n".join(f"[{name}]: {ans}" for name, ans in zip(agents_to_call, results))
    messages = [
        {"role": "system", "content": "Combine the following specialist answers into one concise response."},
        {"role": "user", "content": f"Question: {question}\n\n{combined_context}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Interactive loop ───────────────────────────────────

if __name__ == "__main__":
    print("Campus Assistant — Multi-Agent version (type 'quit' to exit)")
    print("=" * 60)
    print("\nA router agent delegates to two specialists:")
    print("  [course_agent]       → courses, professors, office hours")
    print("  [grade_events_agent] → grade calculations, campus events")
    print("\nExample questions:")
    print("  - When is CSC 446 and where?")
    print("  - What are Dr. Toscano's office hours?")
    print("  - What's happening on campus on 2026-04-18?")
    print("  - I got 85 on the midterm (40%) and 92 on the final (60%). What's my grade?")
    print("  - When is CSC 446 and what events are on 2026-04-18?")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        result = run_manager(user_input)
        print(f"\nAssistant: {result}")

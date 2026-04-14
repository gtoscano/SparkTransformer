"""
Mini-project A: Campus Assistant Agent (smolagents version)

Same tools as demo-campus-assistant.py but using smolagents for the agent loop.
Compare with the pure transformers version to see what smolagents handles for you.

Run:
  python demo-campus-assistant-smolagents.py
"""

from smolagents import CodeAgent, tool
from smolagents.models import TransformersModel

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

# ── Tools ──────────────────────────────────────────────

@tool
def lookup_course(course_id: str) -> str:
    """Look up a course by its ID and return the title, schedule, and room.

    Args:
        course_id: The course identifier to look up (e.g. 'CSC446').
    """
    info = COURSES.get(course_id.upper().replace(" ", ""))
    if not info:
        return f"No course found for '{course_id}'. Available: {', '.join(COURSES.keys())}."
    return f"{course_id.upper()} — {info['title']}\nSchedule: {info['schedule']}\nRoom: {info['room']}"


@tool
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


@tool
def campus_events(date: str) -> str:
    """Return campus events for a given date (YYYY-MM-DD format).

    Args:
        date: The date to check in YYYY-MM-DD format (e.g. '2026-04-18').
    """
    events = EVENTS.get(date)
    if not events:
        return f"No events found for {date}."
    return f"Events on {date}:\n" + "\n".join(f"  - {e}" for e in events)


@tool
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


# ── Agent ──────────────────────────────────────────────

model = TransformersModel(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    device_map="auto",
)

agent = CodeAgent(
    tools=[lookup_course, office_hours, campus_events, grade_calculator],
    model=model,
    instructions=(
        "You are a helpful campus assistant for The Catholic University of America. "
        "Use the available tools to answer questions about courses, professor office hours, "
        "campus events, and grade calculations. "
        "If you don't need a tool, answer directly. Be concise. "
        "Always combine all results into a single final_answer call."
    ),
)

# ── Interactive loop ───────────────────────────────────

if __name__ == "__main__":
    print("Campus Assistant — smolagents version (type 'quit' to exit)")
    print("=" * 55)
    print("\nExample questions you can ask:")
    print("  - When is CSC 446 and where?")
    print("  - What are Dr. Toscano's office hours?")
    print("  - What's happening on campus on 2026-04-18?")
    print("  - I got 85 on the midterm (40%) and 92 on the final (60%). What's my grade?")
    print("  - Who teaches in Pangborn 312?")
    print("=" * 55)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        result = agent.run(user_input)
        print(f"\nAssistant: {result}")

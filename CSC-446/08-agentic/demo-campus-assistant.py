"""
Mini-project A: Campus Assistant Agent

A local LLM agent with four tools:
  - lookup_course(course_id)
  - office_hours(professor)
  - campus_events(date)
  - grade_calculator(scores, weights)

Run locally:
  python demo-campus-assistant.py

Run via Telegram:
  export TELEGRAM_BOT_TOKEN="your-token-here"
  python demo-campus-assistant.py --telegram
"""

import json
import re
import sys
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

# ── Tools ──────────────────────────────────────────────
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


def campus_events(date: str) -> str:
    """Return campus events for a given date (YYYY-MM-DD format).

    Args:
        date: The date to check in YYYY-MM-DD format (e.g. '2026-04-18').
    """
    events = EVENTS.get(date)
    if not events:
        return f"No events found for {date}."
    return f"Events on {date}:\n" + "\n".join(f"  • {e}" for e in events)


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


TOOLS = {
    "lookup_course": lookup_course,
    "office_hours": office_hours,
    "campus_events": campus_events,
    "grade_calculator": grade_calculator,
}

tool_functions = [lookup_course, office_hours, campus_events, grade_calculator]

SYSTEM_PROMPT = (
    "You are a helpful campus assistant for The Catholic University of America. "
    "Use the available tools to answer questions about courses, professor office hours, "
    "campus events, and grade calculations. "
    "If you don't need a tool, answer directly. Be concise."
)

# ── Agent loop ─────────────────────────────────────────

def run_agent(user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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

            if tool_name not in TOOLS:
                tool_result = f"Error: unknown tool '{tool_name}'."
            else:
                tool_result = TOOLS[tool_name](**arguments)

            print(f"  [tool] {tool_name}({arguments}) → {tool_result}")

            messages.append({"role": "tool", "name": tool_name, "content": tool_result})

    return response


# ── CLI mode ───────────────────────────────────────────

def run_cli():
    print("Campus Assistant (type 'quit' to exit)")
    print("=" * 45)
    print("\nExample questions you can ask:")
    print("  - When is CSC 446 and where?")
    print("  - What are Dr. Toscano's office hours?")
    print("  - What's happening on campus on 2026-04-18?")
    print("  - I got 85 on the midterm (40%) and 92 on the final (60%). What's my grade?")
    print("  - Who teaches in Pangborn 312?")
    print("=" * 45)
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        reply = run_agent(user_input)
        print(f"\nAssistant: {reply}")


# ── Telegram mode ──────────────────────────────────────

def run_telegram():
    import os
    from telegram import Update
    from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN env var first.")

    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_text = update.message.text
        reply = run_agent(user_text)
        await update.message.reply_text(reply)

    app = ApplicationBuilder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Campus Assistant Telegram bot is running.")
    app.run_polling()


# ── Main ───────────────────────────────────────────────

if __name__ == "__main__":
    if "--telegram" in sys.argv:
        run_telegram()
    else:
        run_cli()

"""
Telegram bot that wraps the local LLM agent loop from demo1.py.

Setup:
  1. pip install python-telegram-bot transformers torch
  2. Message @BotFather on Telegram → /newbot → copy the token
  3. export TELEGRAM_BOT_TOKEN="your-token-here"
  4. python demo-telegram.py
"""

import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# ── Model ──────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype="auto",
)

# ── Tools ──────────────────────────────────────────────
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '12*(3+4)'.

    Args:
        expression: The arithmetic expression to evaluate.
    """
    allowed = "0123456789+-*/(). "
    if any(ch not in allowed for ch in expression):
        return "Error: unsupported characters in expression."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


def get_student_name(course_id: str) -> str:
    """Return a mock student roster summary for a course ID.

    Args:
        course_id: The course identifier to look up (e.g. 'CSC101').
    """
    fake_db = {
        "CSC101": "Alice, Bob, Carla",
        "CSC202": "Diego, Emma, Farah",
        "LLM500": "Grace, Henry, Isabel",
    }
    return fake_db.get(course_id.upper(), "No roster found for that course.")


TOOLS = {"calculator": calculator, "get_student_name": get_student_name}
tool_functions = [calculator, get_student_name]

# ── Agent loop (same logic as demo1.py) ────────────────
def run_agent(user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful teaching assistant. Use tools when needed."},
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

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "tool", "name": tool_name, "content": tool_result})

    return response


# ── Telegram handler ───────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    reply = run_agent(user_text)
    await update.message.reply_text(reply)


# ── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN env var first. See docstring.")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running — send a message on Telegram.")
    app.run_polling()

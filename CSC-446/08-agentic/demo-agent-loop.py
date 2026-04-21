import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # replace with your preferred local model

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype="auto",
)

# -----------------------------
# 1. Define local tools
# -----------------------------
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

TOOLS = {
    "calculator": calculator,
    "get_student_name": get_student_name,
}

tool_functions = [calculator, get_student_name]

# -----------------------------
# 2. Initial conversation
# -----------------------------
# Possible questions to try:
#
# Uses calculator only:
#   "What is 12*(3+4)?"
#   "Compute 100/4 + 25*2."
#   "How much is (99+1)*(50-25)?"
#
# Uses get_student_name only:
#   "Who is enrolled in CSC101?"
#   "Give me the roster for CSC202."
#   "List the students in LLM500."
#
# Uses both tools:
#   "How many students are listed in LLM500, and what is 12*(3+4)?"
#   "Who is in CSC101, and what is 7*8?"
#
# Uses no tools (answered from general knowledge):
#   "What does NLP stand for?"
#   "Explain what a transformer is."
#
# Tool returns no results (unknown course):
#   "Who is enrolled in MATH999?"
#
messages = [
    {"role": "system", "content": "You are a helpful teaching assistant. Use tools when needed."},
    {"role": "user", "content": "How many students are listed in LLM500, and what is 12*(3+4)?"}
]

# -----------------------------
# 3. Agent loop
# -----------------------------
for step in range(5):
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tool_functions,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\n=== MODEL RAW OUTPUT (step {step+1}) ===")
    print(response)

    # ----------------------------------------
    # 4. Parse tool calls from the model output
    # ----------------------------------------
    # Qwen-style models wrap tool calls in <tool_call>...</tool_call> tags.
    # Each tag contains JSON with "name" and "arguments".
    # A response may contain multiple tool calls.
    # If no tool calls are found, we treat the response as the final answer.

    import re
    tool_calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)

    if not tool_calls:
        print("\n=== FINAL ANSWER ===")
        print(response)
        break

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
            tool_fn = TOOLS[tool_name]
            tool_result = tool_fn(**arguments)

        print(f"\n=== TOOL CALL ===")
        print(tool_name, arguments)
        print("TOOL RESULT:", tool_result)

        messages.append({
            "role": "tool",
            "name": tool_name,
            "content": tool_result,
        })

from smolagents import CodeAgent, tool
from smolagents.models import TransformersModel

@tool
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

@tool
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

model = TransformersModel(
    model_id="Qwen/Qwen2.5-3B-Instruct",   # replace if needed
    device_map="auto",
)

agent = CodeAgent(
    tools=[calculator, get_student_name],
    model=model,
    additional_authorized_imports=[],
    instructions=(
        "You are a helpful teaching assistant. Use the provided tools when needed. "
        "Always combine all results into a single final_answer call. "
        "Never call final_answer more than once."
    ),
)

result = agent.run("How many students are listed in LLM500, and what is 12*(3+4)?")
print(result)

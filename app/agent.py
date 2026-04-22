import os
import json
import re
from datetime import datetime, timedelta
from groq import Groq
from groq import RateLimitError, GroqError
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


tools = [
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a clinical appointment for a patient",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full name of the patient"
                    },
                    "doctor_name": {
                        "type": "string",
                        "description": "Name of the doctor"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date of appointment e.g. 2024-01-15"
                    },
                    "time_slot": {
                        "type": "string",
                        "description": "Time of appointment e.g. 10:00 AM"
                    }
                },
                "required": ["patient_name", "doctor_name", "date", "time_slot"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check if a doctor is available on a given date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {
                        "type": "string",
                        "description": "Name of the doctor"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date to check e.g. 2024-01-15"
                    },
                    "time_slot": {
                        "type": "string",
                        "description": "Time slot to check e.g. 10:00 AM"
                    }
                },
                "required": ["doctor_name", "date", "time_slot"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an existing appointment",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full name of the patient"
                    },
                    "doctor_name": {
                        "type": "string",
                        "description": "Name of the doctor"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date of the appointment to cancel"
                    }
                },
                "required": ["patient_name", "doctor_name", "date"]
            }
        }
    }
]


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    This is where tool calls get executed.
    Later we will connect this to the real database.
    For now it returns mock responses.
    """
    if tool_name == "book_appointment":
        return json.dumps({
            "status": "success",
            "message": f"Appointment booked for {tool_args['patient_name']} "
                      f"with {tool_args['doctor_name']} on "
                      f"{tool_args['date']} at {tool_args['time_slot']}",
            "appointment_id": "APT-001"
        })

    elif tool_name == "check_availability":
        doctor_name = format_doctor_name(tool_args['doctor_name'])
        return json.dumps({
            "status": "available",
            "doctor": doctor_name,
            "date": tool_args['date'],
            "time_slot": tool_args['time_slot'],
            "message": f"{doctor_name} is available at {tool_args['time_slot']}"
        })

    elif tool_name == "cancel_appointment":
        return json.dumps({
            "status": "cancelled",
            "message": f"Appointment cancelled for {tool_args['patient_name']} "
                      f"with {tool_args['doctor_name']} on {tool_args['date']}"
        })

    return json.dumps({"status": "error", "message": "Unknown tool"})


def format_doctor_name(doctor_name: str) -> str:
    cleaned_name = doctor_name.strip()
    if cleaned_name.lower().startswith("dr."):
        return cleaned_name
    return f"Dr. {cleaned_name}"


def parse_local_intent(user_message: str) -> tuple[str, dict] | tuple[None, None]:
    text = user_message.lower()
    patient_name = extract_patient_name(user_message)
    doctor_name = extract_doctor_name(user_message)

    if any(keyword in text for keyword in ["book", "appointment", "schedule", "reserve"]):
        date = extract_date(text)
        time_slot = extract_time_slot(text)
        if patient_name and doctor_name and date and time_slot:
            return "book_appointment", {
                "patient_name": patient_name,
                "doctor_name": doctor_name,
                "date": date,
                "time_slot": time_slot,
            }

    if any(keyword in text for keyword in ["available", "availability", "free"]):
        date = extract_date(text)
        time_slot = extract_time_slot(text)
        if doctor_name and date and time_slot:
            return "check_availability", {
                "doctor_name": doctor_name,
                "date": date,
                "time_slot": time_slot,
            }

    if any(keyword in text for keyword in ["cancel", "delete", "remove"]):
        date = extract_date(text)
        if patient_name and doctor_name and date:
            return "cancel_appointment", {
                "patient_name": patient_name,
                "doctor_name": doctor_name,
                "date": date,
            }

    return None, None


def extract_patient_name(user_message: str) -> str | None:
    match = re.search(
        r"(?:my name is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        user_message,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def extract_doctor_name(user_message: str) -> str | None:
    doctor_match = re.search(
        r"\bdr\.?\s+([A-Za-z]+(?:\s+(?!tomorrow\b|today\b|on\b|at\b|available\b|availability\b|is\b|was\b|for\b|with\b|my\b|the\b)[A-Za-z]+)?)"
        r"(?=\s+(?:tomorrow|today|on|at|available|availability|is|was|for|with|my|the)\b|[?.!,]|$)",
        user_message,
        flags=re.IGNORECASE,
    )
    if doctor_match:
        doctor_name = doctor_match.group(1).strip()
        return format_doctor_name(doctor_name.title())

    with_match = re.search(
        r"\bwith\s+([A-Za-z]+(?:\s+(?!tomorrow\b|today\b|on\b|at\b|available\b|availability\b|is\b|was\b|for\b|my\b|the\b)[A-Za-z]+)?)"
        r"(?=\s+(?:tomorrow|today|on|at|available|availability|is|was|for|my|the)\b|[?.!,]|$)",
        user_message,
        flags=re.IGNORECASE,
    )
    if with_match:
        return format_doctor_name(with_match.group(1).strip().title())

    return None


def extract_date(text: str) -> str | None:
    today = datetime.now().date()
    if "tomorrow" in text:
        return (today + timedelta(days=1)).isoformat()

    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    return match.group(1) if match else None


def extract_time_slot(text: str) -> str | None:
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if not match:
        return None

    hour = int(match.group(1))
    minutes = match.group(2) or "00"
    meridiem = match.group(3).upper()
    return f"{hour}:{minutes} {meridiem}"


# ─────────────────────────────────────────
# MAIN AGENT — takes user text, thinks,
# calls tools if needed, returns response
# ─────────────────────────────────────────
def run_agent(user_message: str, conversation_history: list = []) -> str:
    """
    The core agent loop:
    1. Send user message to GPT-4o
    2. If GPT wants to call a tool → execute it
    3. Send tool result back to GPT
    4. Return final response
    """

    # System prompt — tells the LLM who it is and what it does
    system_prompt = """You are a helpful clinical appointment booking assistant 
    for a healthcare platform. You help patients book, reschedule, and cancel 
    appointments with doctors. You support English, Hindi, and Tamil languages.
    Always respond in the same language the patient uses.
    Be polite, clear, and concise."""

    # Build messages list with history for context
    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user", "content": user_message}
    ]

    print(f"\n[AGENT] User said: {user_message}")

    # Step 1 — Send to GPT-4o
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"  # LLM decides when to use tools
        )
    except (RateLimitError, GroqError) as exc:
        print(f"[AGENT] Groq API error: {exc}")
        fallback_result = run_local_fallback(user_message)
        print(f"[AGENT] Fallback response: {fallback_result}")
        return fallback_result

    message = response.choices[0].message

    # Step 2 — Check if LLM wants to call a tool
    if message.tool_calls:
        print(f"[AGENT] LLM decided to call tool: {message.tool_calls[0].function.name}")

        # Add LLM's decision to messages
        messages.append(message)

        # Execute each tool the LLM requested
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"[AGENT] Executing: {tool_name} with args: {tool_args}")

            # Actually run the tool
            tool_result = execute_tool(tool_name, tool_args)

            print(f"[AGENT] Tool result: {tool_result}")

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # Step 3 — Send tool results back to GPT for final response
        final_response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=tools,
        )

        final_message = final_response.choices[0].message.content
        print(f"[AGENT] Final response: {final_message}")
        return final_message

    # No tool needed — direct response
    direct_response = message.content
    print(f"[AGENT] Direct response: {direct_response}")
    return direct_response


def run_local_fallback(user_message: str) -> str:
    tool_name, tool_args = parse_local_intent(user_message)
    if not tool_name:
        return (
            "I cannot reach the LLM right now because the Groq API is unavailable, "
            "and I could not infer a supported appointment action locally."
        )

    print(f"[AGENT] Local fallback selected tool: {tool_name}")
    print(f"[AGENT] Local fallback args: {tool_args}")
    tool_result = execute_tool(tool_name, tool_args)
    tool_data = json.loads(tool_result)
    return tool_data.get("message", tool_result)



if __name__ == "__main__":
    print("=" * 50)
    print("Testing LLM Agent with tool calling...")
    print("=" * 50)

    # Test 1 — Book an appointment
    response1 = run_agent(
        "I want to book an appointment with Dr. Sharma tomorrow at 10am. My name is Ravi Kumar."
    )
    print(f"\nFinal answer to user: {response1}")

    print("\n" + "=" * 50)

    # Test 2 — Check availability
    response2 = run_agent(
        "Is Dr. Sharma available on 2024-01-20 at 2pm?"
    )
    print(f"\nFinal answer to user: {response2}")
from dotenv import load_dotenv
import streamlit as st
import os
import asyncio
from pydantic import BaseModel
from agents import (
    Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel,
    RunContextWrapper, GuardrailFunctionOutput, InputGuardrailTripwireTriggered,
    input_guardrail
)

# ========== Load Gemini API ==========
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# ========== Setup Gemini client ==========
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# ========== Menu & Prices ==========
MENU = {
    "small": 1200,
    "medium": 1700,
    "large": 2200,
    "extra cheese": 200,
    "mushrooms": 150,
    "pepperoni": 250,
    "coke": 100
}

# Sidebar menu
st.sidebar.title("🍽️ Menu & Prices")
for item, price in MENU.items():
    st.sidebar.write(f"**{item.capitalize()}** — Rs {price}")

# ========== Output Schemas ==========
class PizzaCheckOutput(BaseModel):
    is_pizza_order: bool
    reasoning: str

class PizzaOrderOutput(BaseModel):
    size: str | None
    toppings: list[str] | None
    quantity: int | str | None
    drinks: list[str] | None

# ========== Classifier Agent ==========
classifier_agent = Agent(
    name="Pizza Classifier Agent",
    instructions=(
        "Determine if the user is placing a pizza order. "
        "If not, respond ONLY in JSON with {is_pizza_order: false, reasoning: 'reason'}. "
        "If yes, respond {is_pizza_order: true, reasoning: 'reason'}."
    ),
    output_type=PizzaCheckOutput
)

# ========== Guardrail ==========
@input_guardrail
async def non_pizza_guardrail(context: RunContextWrapper, agent: Agent, user_input: str):
    result = await Runner.run(
        classifier_agent,
        input=user_input,
        run_config=config,
        context=context.context
    )
    is_pizza = result.final_output.is_pizza_order
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not is_pizza
    )

# ========== Pizza Agent ==========
pizza_agent = Agent(
    name="Pizza Order Agent",
    instructions=(
        "Extract details from user input and return ONLY JSON with: "
        "size (string), toppings (list), quantity (int), drinks (list). "
        "If quantity is in words (one, two, three), convert to a number. "
        "If anything is missing, return null for that field."
    ),
    input_guardrails=[non_pizza_guardrail],
    output_type=PizzaOrderOutput
)

# ========== Response Agent (Final Confirmation Only) ==========
response_agent = Agent(
    name="Pizza Confirmation Agent",
    instructions=(
        "You are a polite pizza assistant. "
        "Confirm the order confidently. Never ask questions. "
        "Respond in one final sentence with total price and delivery time (30–40 minutes). "
        "Use emojis like ✅ or 🍕."
    ),
)

# ========== Helpers ==========
word_to_num = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10
}

def fix_quantity(q):
    # Accept ints, numeric strings, and word numbers; fallback to 1
    try:
        if q is None:
            return 1
        if isinstance(q, int):
            return max(1, q)
        if isinstance(q, float):
            return max(1, int(q))
        if isinstance(q, str):
            q_clean = q.strip().lower()
            if q_clean.isdigit():
                return max(1, int(q_clean))
            if q_clean in word_to_num:
                return word_to_num[q_clean]
        # unexpected type -> fallback
        return 1
    except Exception:
        return 1

def fix_size(size):
    # Accept None, strings; fallback to 'medium'
    try:
        if not size:
            return "medium"
        if not isinstance(size, str):
            return "medium"
        s = size.strip().lower()
        # accept common synonyms
        if s in MENU:
            return s
        if s.startswith("s"):  # small
            return "small"
        if s.startswith("m"):
            return "medium"
        if s.startswith("l"):
            return "large"
        return "medium"
    except Exception:
        return "medium"

def normalize_list(lst):
    if not lst:
        return []
    # sometimes model returns single string - handle that
    if isinstance(lst, str):
        return [lst]
    if isinstance(lst, list):
        return [str(x).strip().lower() for x in lst if str(x).strip()]
    return []

def calculate_price(size, toppings, drinks, quantity):
    total = MENU.get(size.lower(), 0)
    for t in toppings:
        total += MENU.get(t.lower(), 0)
    for d in drinks:
        total += MENU.get(d.lower(), 0)
    return total * quantity

# ========== Streamlit UI ==========
st.title("🍕 AI Pizza Order Assistant")

async def run_agent(user_input):
    """
    Always return a 6-tuple:
    (final_text, size, toppings, drinks, quantity, total_price)
    On non-pizza or errors, size will be None and totals 0.
    """
    try:
        # Step 1: Extract order (guardrail will run before pizza_agent if input is non-pizza)
        result = await Runner.run(
            pizza_agent,
            input=user_input,
            run_config=config,
        )
        output = result.final_output

        # normalize / fix values
        size = fix_size(output.size)
        toppings = normalize_list(output.toppings)
        drinks = normalize_list(output.drinks)
        quantity = fix_quantity(output.quantity)

        # calculate price
        total_price = calculate_price(size, toppings, drinks, quantity)

        # prepare confirmation prompt for response agent
        confirmation_input = (
            f"Order details: {quantity} {size} pizza(s) "
            f"with {', '.join(toppings) if toppings else 'no toppings'} "
            f"and {', '.join(drinks) if drinks else 'no drinks'}. "
            f"Total Rs {total_price}."
        )

        ai_response = await Runner.run(
            response_agent,
            input=confirmation_input,
            run_config=config,
        )
        # response agent might return text or an object; coerce to string
        final_text = ai_response.final_output
        if not isinstance(final_text, str):
            final_text = str(final_text)

        return final_text, size, toppings, drinks, quantity, total_price

    except InputGuardrailTripwireTriggered as e:
        # Non-pizza: return consistent shape with None/empty defaults
        try:
            reason = e.tripwire_output.output_info.reasoning
        except Exception:
            reason = "Input rejected by guardrail."
        msg = f"🚨 Not a pizza order. Reason: {reason}"
        return msg, None, [], [], 0, 0

    except Exception as e:
        # Unexpected errors - return consistent shape and helpful message
        return f"🚨 Error processing order: {str(e)}", None, [], [], 0, 0

user_input = st.text_input("What would you like to order?")

if st.button("Place Order"):
    if user_input.strip():
        st.write("Processing your order...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        (
            final_text,
            size,
            toppings,
            drinks,
            quantity,
            total_price
        ) = loop.run_until_complete(run_agent(user_input))
        loop.close()

        # If size is None -> non-pizza or error
        if size:
            st.success(final_text)

            # Order Summary
            st.markdown("### 🧾 Order Summary")
            st.write(f"**🍕 Size:** {size.capitalize()}")
            st.write(f"**🧂 Toppings:** {', '.join(toppings) if toppings else 'None'}")
            st.write(f"**🥤 Drinks:** {', '.join(drinks) if drinks else 'None'}")
            st.write(f"**🔢 Quantity:** {quantity}")
            st.write(f"**💰 Total Price:** Rs {total_price}")
            st.info("🚚 Your order will arrive in 30–40 minutes. Thank you!")
        else:
            st.error(final_text)
    else:
        st.warning("Please enter your order.")

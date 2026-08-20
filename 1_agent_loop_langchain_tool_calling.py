from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

from langchain.tools import tool

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langsmith import traceable

MAX_ITERATION = 10
MODEL = "qwen3:1.7b"


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in catalog."""
    print(f"Executing get_product_price...")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}

    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """apply discount tier to the price, avaliable tiers: gold, silver, bronze"""
    print(f"Executing apply_discount...")
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentage.get(discount_tier, 0)

    return round(price * (1 - discount / 100), 2)


# Agents Loop
@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

    llm_with_tools = llm.bind_tools(tools)
    print("=" * 60)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant"
                "you have access to a product catalog tool"
                "and a discount tool. \n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price"
                "2. Only call the applied_discount AFTER you have received the price from get_product_price"
                "DONOT pass a made-up number .\n"
                "3. NEVER caclculate discounts yourself using math."
                "Always use the apply_discount tool \n"
                "4. If the user does not specify a dicount tier"
                "ask them which tier to use - DO NOT assume one"
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(0, MAX_ITERATION):
        print(f"---Iteration {iteration} -----")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\n Final answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"[TOOL result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print(f"ERROR: MAx oteration without a final answer")
    return None


if __name__ == "__main__":
    print("Hello langchain agent!")
    print()
    result = run_agent("what is the price of a laptop after applying a gold discount?")

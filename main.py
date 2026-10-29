from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from chains import generate_chain, reflect_chain

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessagesState):
    response = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def reflection_node(state: MessagesState):
    response = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}


builder = StateGraph(MessagesState)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.add_edge(START, GENERATE)


def should_continue(state: MessagesState):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())

def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)

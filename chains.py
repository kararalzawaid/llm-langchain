from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral Twitter influencer grading a tweet. Generate critique and "
            "recommendations for the user's tweet. Always provide detailed "
            "recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Twitter techie influencer assistant tasked with writing excellent "
            "Twitter posts. Generate the best Twitter post possible for the user's "
            "requests. If the user provides critique, respond with a revised version of "
            "your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# ChatOpenAI's default in this setup resolves to the retired gpt-3.5-turbo.
llm = ChatOpenAI(model="gpt-4o-mini")

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, human
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings,
)

# only three results are returned
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context: {context}\n\nQuestion: {question}\nAnswer:"
)


def format_docs(docs):
    """Format the documents to be used as context for the question-answering chain."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_whithout_lcel(query):
    """Retrieval chain without LangChain Expressions."""
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)

    return response.content


def retrieval_chain_with_lcel():
    """Retrieval chain with LangChain Expressions."""

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("retrieving documents...")

    query = "What is pinecone in machine learning?"

    # print("\n" + "=" *70)
    # print("IMPLEMENTATION 0: Raw LLM Invocation (No Rag)")
    # print("\n" + "=" *70)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print('\n Answer:')
    # print(result_raw.content)

    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 1: Retrieval Chain (No LangChain Expressions)")
    # print("\n" + "=" * 70)
    # result_retrieval = retrieval_chain_whithout_lcel(query)
    # print("\n Answer:")
    # print(result_retrieval)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: with LCEL - Better approach")
    print("\n" + "=" * 70)
    chain_with_lcel = retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print(result_with_lcel)

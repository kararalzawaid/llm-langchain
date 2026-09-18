import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    print(os.environ)
    loader = TextLoader("./mediumblog1.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(len(chunks))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    print("ingesting data into pinecone...")
    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=os.environ["INDEX_NAME"]
    )

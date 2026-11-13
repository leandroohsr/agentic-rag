from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma

from langchain_ollama import OllamaEmbeddings

load_dotenv()


# Web pages used as the knowledge base
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Download each page and convert it into Documents (text + metadata)
docs = []

for url in urls:
    docs.extend(UnstructuredLoader(web_url=url, chunking_strategy="basic", max_characters=1000000).load())


# print(docs[0])


# Splitter that breaks text into chunks of up to 250 tokens, with no overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250, chunk_overlap = 0
)


# Apply the splitter to produce the chunks
doc_splits = text_splitter.split_documents(docs)


# Embed the documents and store them in the Chroma vector database
# vectorstore = Chroma.from_documents(
#     documents = doc_splits,
#     collection_name="rag-chroma",
#     embedding = OllamaEmbeddings(model="nomic-embed-text"),
#     persist_directory="./.chroma"
# )

# Create a retriever from the vectorstore
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
).as_retriever(search_kwargs={"k": 4})


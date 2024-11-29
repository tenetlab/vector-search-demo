import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()  # take environment variables from .env.

openai_key = os.getenv("OPENAI_API_KEY")

print("OpenAI key:", openai_key)
print("=================================")

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('raw_text.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())


query = "what did Putin do?"

# Similarity search
docs = db.similarity_search(query)
print(docs[0].page_content)
print("=================================")

# Similarity search by vector
embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)
print("=================================")
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os

# Set up
docs_dir = "./expert_knowledge"
collection_name = "knowledge_base"

# Connect to Qdrant
qdrant = QdrantClient(url="http://localhost:6333")
if collection_name not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# Load and split
all_docs = []
for file in os.listdir(docs_dir):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(docs_dir, file))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        all_docs.extend(splitter.split_documents(docs))

# Embed and store
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Qdrant(
    client=qdrant,
    collection_name=collection_name,
    embeddings=embeddings
)
vectorstore.add_documents(all_docs)

print(f"✅ {len(all_docs)} documents embedded to {collection_name}")

# Install necessary packages
install -q langchain langchain-community langchain-qdrant fastembed qdrant-client pypdf

import os
import gc
import time
from google.colab import userdata
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ==========================================
# CONFIGURATION (SECURE)
# ==========================================
# Make sure your secrets in Colab are named exactly 'QDRANT_URL' and 'QDRANT_API_KEY'
try:
    QDRANT_URL = userdata.get('QDRANT_URL')
    QDRANT_API_KEY = userdata.get('QDRANT_API_KEY')
except Exception as e:
    print("❌ Error fetching secrets. Please ensure you added 'QDRANT_URL' and 'QDRANT_API_KEY' in the Key icon on the left sidebar.")
    raise e

COLLECTION_NAME = "RAG_Netflix_se"

import os

if os.path.exists('/content/drive'):
    print("✅ Yes, Google Drive is mounted!")
    print("Files visible:", os.listdir('/content/drive/MyDrive')[:5]) # Lists first 5 files to verify
else:
    print("❌ No, Drive is NOT mounted.")
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Drive mounted successfully.")

# Path to your dataset on Google Drive
ROOT_DIRECTORY = "/content/drive/MyDrive/syllabus_pdfs"

# Connect to Qdrant Cloud
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create Collection if it doesn't exist (Vector Size 768 for Jina Base)
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print(f"✅ Created collection: {COLLECTION_NAME}")
else:
    print(f"✅ Connected to existing collection: {COLLECTION_NAME}")


# FastEmbed (Jina-v2-base-en) - Handles 8k context, very efficient
embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")

# ==========================================
# 3. THE SAFE INGESTION LOOP
# ==========================================
print("\n🚀 Starting Safe Ingestion Loop...")

file_count = 0
error_count = 0

# Walk through every folder and file
for root, dirs, files in os.walk(ROOT_DIRECTORY):
    for file in files:
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(root, file)
            file_count += 1

            try:
                # --- A. Extract Metadata ---
                meta = extract_metadata_from_path(file_path)
                print(f"\nProcessing [{file_count}]: {file} | Subject: {meta['subject']}")

                # --- B. Load PDF ---
                loader = PyPDFLoader(file_path)
                # Load and split immediately to avoid holding full PDF in RAM
                chunks = loader.load_and_split(text_splitter)

                # --- C. Inject Metadata ---
                # We add the folder info to EVERY chunk so the Router can find it later
                for chunk in chunks:
                    chunk.metadata.update(meta)

                # --- D. Upload to Qdrant ---
                # This sends vectors to the cloud
                vector_store.add_documents(chunks)
                print(f"   👉 Uploaded {len(chunks)} chunks to Cloud.")

                # --- E. MEMORY CLEANUP (The Magic Trick) ---
                del chunks
                del loader
                gc.collect() # Force Python to release RAM immediately

                # Small sleep to be nice to the API
                time.sleep(0.5)

            except Exception as e:
                print(f"   ❌ Error processing {file}: {e}")
                error_count += 1
                # Even if it fails, clean up RAM
                gc.collect()

print(f"\n🎉 Ingestion Complete!")
print(f"Total Files Scanned: {file_count}")
print(f"Errors: {error_count}")


# Force install the specific missing library and its friends
!pip install -U langchain-google-genai langchain langchain-community google-generativeai qdrant-client fastembed
!pip install langchain_qdrant


import os
from google.colab import userdata
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

# ==========================================
# 1. CONFIGURATION & CONNECTION
# ==========================================
try:
    QDRANT_URL = userdata.get('QDRANT_URL')
    QDRANT_API_KEY = userdata.get('QDRANT_API_KEY')
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY') # New Key!
except:
    print("❌ Missing Secrets! Ensure QDRANT_URL, QDRANT_API_KEY, and GOOGLE_API_KEY are set.")

COLLECTION_NAME = "RAG_Netflix_se"

# Initialize the LLM (The Brain)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Initialize Embeddings (Must be same model as Phase 1)
embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")

# Connect to Qdrant Cloud
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
print("✅ Connected to Qdrant Cloud.")


import os
from google.colab import userdata
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ==========================================
# 1. CONFIGURATION
# ==========================================
try:
    QDRANT_URL = userdata.get('QDRANT_URL')
    QDRANT_API_KEY = userdata.get('QDRANT_API_KEY')
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    COLLECTION_NAME = "RAG_Netflix_se"
except Exception as e:
    print("❌ Keys missing! Check Secrets.")

# ==========================================
# 2. INITIALIZATION
# ==========================================
print("🚀 Initializing RAG System with Citations...")

embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

print(f"✅ System Ready. Connected to '{COLLECTION_NAME}'.")

# ==========================================
# 3. RETRIEVER WITH SOURCE EXTRACTION
# ==========================================
def retrieve_context_with_sources(query):
    """
    Searches database and returns:
    1. Combined text content (for the AI)
    2. List of unique sources (for the User)
    """
    print(f"\n🔎 Searching docs for: '{query}'...")
    try:
        # Get top 5 chunks
        docs = vector_store.similarity_search(query, k=5)

        if not docs:
            return "", []

        # Combine text for the AI
        context = "\n\n".join([d.page_content for d in docs])

        # Extract unique sources for the user
        # We try multiple keys in case 'source' is missing
        sources = set()
        for d in docs:
            meta = d.metadata
            # Try to find a valid name in this order: source -> title -> file_name -> Unknown
            src = meta.get('source') or meta.get('title') or meta.get('file_name') or "Unknown File"

            # Clean up the path (e.g., /content/drive/My Drive/.../textbook.pdf -> textbook.pdf)
            if "/" in str(src):
                src = str(src).split("/")[-1]

            sources.add(src)

        print(f"   📄 Found {len(docs)} relevant segments from {len(sources)} file(s).")
        return context, list(sources)

    except Exception as e:
        print(f"   ❌ Retrieval Error: {e}")
        return "", []

# ==========================================
# 4. GENERATOR
# ==========================================
PROMPTS = {
    "Quick": "You are a revision assistant. Define in <60 words. CONTEXT: {context} QUERY: {query} ANSWER:",
    "Detailed": "You are a Professor. Explain in depth with headings and examples. CONTEXT: {context} QUERY: {query} ANSWER:",
    "Exam": "You are a top student. Structure as: 1. Definition 2. Types/Formula 3. Application. CONTEXT: {context} QUERY: {query} ANSWER:"
}

def ask_tutor(query, mode="Detailed"):
    # 1. Retrieve Content AND Sources
    context_text, sources = retrieve_context_with_sources(query)

    if not context_text:
        return "⚠️ No info found.", []

    # 2. Generate Answer
    print(f"   ⚙️ Generating [{mode}] answer...")
    template = PROMPTS.get(mode, PROMPTS["Detailed"])
    prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{query}")])
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"context": context_text, "query": query})

    return response, sources

# ==========================================
# 5. FINAL TEST
# ==========================================
query = "WHO IS VIRAT"  # <--- Change this to test!
mode = "Quick"

# Run
answer, cited_sources = ask_tutor(query, mode)

print("\n" + "="*50)
print(f"🤖 AI TUTOR RESPONSE ({mode} Mode)")
print("="*50)
print(answer)

print("\n" + "-"*50)
print("📚 SOURCES USED:")
if cited_sources:
    for i, source in enumerate(cited_sources, 1):
        print(f"{i}. 📂 {source}")
else:
    print("   (Metadata missing in database)")
print("-"*50)
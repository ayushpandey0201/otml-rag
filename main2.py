import os
import nltk
import streamlit as st
import tempfile
from dotenv import load_dotenv
from typing import List, Tuple, Set

# Download necessary NLTK data
def download_nltk_data():
    resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except (LookupError, OSError):
            nltk.download(resource, quiet=True)
download_nltk_data()

# Integration Imports
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Google Drive Imports (Auth needs to be handled carefully)
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Load Environment Variables
load_dotenv()

# --- CONFIGURATION ---
page_title = "RAG Assistant"
page_icon = "🤖"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

# Constants
COLLECTION_NAME = "RAG_Knowledge_Base"
QDRANT_URL = os.getenv("QDRANT_URL")
if QDRANT_URL:
    QDRANT_URL = QDRANT_URL.strip().rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SCOPES = ['https://www.googleapis.com/auth/drive.file'] # Scope for Drive API

# Check API Keys
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env files.")
    st.stop()
if not QDRANT_URL or not QDRANT_API_KEY:
    st.error("❌ Qdrant Credentials not found in .env files.")
    st.stop()
    
# Basic URL format check
if "http" not in QDRANT_URL:
    st.error(f"❌ Invalid QDRANT_URL: '{QDRANT_URL}'. Must start with http:// or https://")
    st.stop()

# --- QDRANT CONNECTION ---
@st.cache_resource
def get_qdrant_client():
    """Initializes and returns a Qdrant Client."""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {e}")
        return None

def init_collection(client):
    """Creates the collection if it doesn't exist."""
    try:
        if not client.collection_exists(COLLECTION_NAME):
            st.write(f"Creating collection '{COLLECTION_NAME}'...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            st.toast(f"Created new collection: {COLLECTION_NAME}", icon="✅")
        return True
    except Exception as e:
        st.error(f"Collection check failed: {e}")
        return False

# --- GOOGLE DRIVE AUTHENTICATION ---
def get_drive_service():
    """Authenticates with Google Drive and returns the service service."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                st.warning("⚠️ `credentials.json` not found. Google Drive upload disabled.")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        st.error(f"Unable to connect to Drive: {e}")
        return None

# --- UI STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    /* Global Dark Theme */
    .stApp {
        background-color: #0E1117; /* Deep Dark Background */
        color: #E0E0E0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* General Text */
    p, span, label, .stMarkdown, div[data-testid="stCaptionContainer"] {
        color: #B0B8C4 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Chat Messages - Dark Glass */
    div[data-testid="stChatMessage"] {
        background-color: #1F2937; /* Clean dark grey */
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* User Message Difference */
    div[data-testid="stChatMessage"][data-testid="user"] {
        background-color: #262730;
    }
    
    /* Primary Action Buttons */
    .stButton>button {
        background: #238636; /* GitHub Green-ish */
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        background-color: #0D1117;
        color: #E0E0E0;
        border-radius: 6px;
        border: 1px solid #30363D;
    }
    .stTextInput>div>div>input:focus {
        border-color: #58A6FF;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #161B22;
        color: #E0E0E0 !important;
        border-radius: 6px;
    }
    
    /* Status Containers */
    div[data-testid="stStatusWidget"] {
        background-color: #1F2937;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def upload_file_to_drive(filepath: str, filename: str):
    """Uploads a file to the root of Google Drive."""
    service = get_drive_service()
    if not service:
        return None
    
    try:
        file_metadata = {'name': filename}
        media = MediaFileUpload(filepath, resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        st.toast(f"Saved to Drive: ID {file.get('id')}", icon="☁️")
        return file.get('id')
    except Exception as e:
        st.error(f"Drive Upload Error: {e}")
        return None

def extract_and_chunk(filepath: str) -> List[Document]:
    """Uses Unstructured to load and chunk the document."""
    try:
        # Load using Unstructured
        loader = UnstructuredFileLoader(filepath)
        docs = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(docs)
        return chunks
    except Exception as e:
        st.error(f"Extraction Error: {e}")
        return []

def process_document(uploaded_file, client):
    """Orchestrates the saving, uploading, extracting, and indexing workflow."""
    status_container = st.status(f"Processing {uploaded_file.name}...", expanded=True)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            # 1. Save locally
            status_container.write("📥 Saving file locally...")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. Upload to Google Drive
            status_container.write("☁️ Uploading to Google Drive...")
            drive_id = upload_file_to_drive(temp_path, uploaded_file.name)
            
            # 3. Extract & Chunk
            status_container.write("📄 Extracting text (this may take a moment)...")
            chunks = extract_and_chunk(temp_path)
            
            if not chunks:
                status_container.update(label="❌ Extraction Failed", state="error")
                return
            
            # 4. Enhance Metadata
            for chunk in chunks:
                chunk.metadata['source'] = uploaded_file.name
                chunk.metadata['drive_id'] = drive_id
                # Try to get page number if Unstructured provided it
                if 'page_number' not in chunk.metadata:
                    chunk.metadata['page_number'] = "Unknown"

            # 5. Embed & Store in Qdrant
            status_container.write("🧠 Generating Embeddings & Indexing...")
            try:
                # Initialize Embeddings (FastEmbed - Jina v2 Base)
                embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
                
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=COLLECTION_NAME,
                    embedding=embeddings,
                )
                
                vector_store.add_documents(chunks)
                st.success(f"✅ Processed {uploaded_file.name} ({len(chunks)} chunks)")
                status_container.update(label="✅ Processing Complete", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Indexing Error: {e}")
                status_container.update(label="❌ Indexing Failed", state="error")
                
    except Exception as e:
        st.error(f"Processing Error: {e}")
        status_container.update(label="❌ Processing Failed", state="error")

# --- RAG ENGINE ---

def get_rag_chain(client):
    """Creates the RAG chain using Gemini and Qdrant."""
    embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    
    template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always cite the document name and page number if available in the context.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(f"Content: {d.page_content}\nSource: {d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page_number', '?')})" for d in docs)
    
    # Custom chain to return both answer and source documents is a bit complex in pure LCEL with just | 
    # So we will do it imperatively in the UI loop for better control over citations display.
    return retriever, llm, prompt


# --- MAIN APP LOOP ---

def main():
    st.title("📚 RAG Assistant with Citations")
    st.caption("Powered by Gemini, Qdrant & Unstructured")
    
    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize Qdrant
    with st.status("🚀 System Initialization...", expanded=True) as status:
        st.write("Checking Qdrant Connection...")
        client = get_qdrant_client()
        if not client:
            status.update(label="❌ Qdrant Connection Failed", state="error")
            return
        
        st.write("Verifying Knowledge Base...")
        if init_collection(client):
            status.update(label="✅ System Ready", state="complete", expanded=False)
        else:
            status.update(label="❌ Collection Error", state="error")
            return
    
    # --- SIDEBAR: Uploads ---
    with st.sidebar:
        st.header("📂 Document Upload")
        uploaded_file = st.file_uploader("Upload PDF, TXT, DOCX", type=["pdf", "txt", "docx", "md"])
        
        if uploaded_file and st.button("Process Document"):
            process_document(uploaded_file, client)
        
        st.divider()
        st.info("Files are uploaded to Google Drive and indexed in Qdrant.")

    # --- CHAT INTERFACE ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("References"):
                    for src in message["sources"]:
                        st.caption(f"📄 {src}")

    if prompt_text := st.chat_input("Ask a question about your documents..."):
        # User Message
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # Assistant Message
        with st.chat_message("assistant"):
            retriever, llm, prompt_template = get_rag_chain(client)
            
            # 1. Retrieval
            with st.spinner("Thinking..."):
                docs = retriever.invoke(prompt_text)
                
                # Context String
                context_str = "\n\n".join([d.page_content for d in docs])
                
                # Chain Execution
                chain = prompt_template | llm | StrOutputParser()
                response = chain.invoke({"context": context_str, "question": prompt_text})
                
                st.markdown(response)
                
                # Extract Sources for UI
                sources = list(set([f"{d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page_number', '?')})" for d in docs]))
                if sources:
                    with st.expander("References"):
                        for src in sources:
                            st.caption(f"📄 {src}")
            
            # Save History
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })

if __name__ == "__main__":
    main()



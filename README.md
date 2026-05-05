# 🚀 RAG System Setup Guide

This guide will help you set up and run the Refactored RAG Assistant.

## 1. Environment Setup

### Install Dependencies
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### Install System Dependencies (for Unstructured)
For PDF and image processing, you need some system libraries:
*   **Mac (Homebrew)**: `brew install tesseract poppler`
*   **Linux**: `sudo apt-get install tesseract-ocr poppler-utils`

## 2. API Keys & Configuration

1.  **Duplicate the template**:
    ```bash
    cp .env.example .env
    ```
2.  **Fill in `.env`**:
    *   `GOOGLE_API_KEY`: Get from [Google AI Studio](https://aistudio.google.com/).
    *   `QDRANT_URL`: Your Qdrant Cloud Cluster URL (e.g., `https://xyz.qdrant.io`).
    *   `QDRANT_API_KEY`: Your Qdrant Cloud API Key.
    *   `UNSTRUCTURED_API_KEY`: (Optional) If you use the hosted API.

## 3. Google Drive Integration (Important) ☁️

To allow the app to upload files to your Google Drive, you need to create OAuth credentials.

1.  Go to **[Google Cloud Console](https://console.cloud.google.com/)**.
2.  Create a new project.
3.  Enable the **Google Drive API**.
4.  Go to **APIs & Services > Credentials** and click **Create Credentials > OAuth client ID**.
5.  Select **Desktop App**.
6.  Download the JSON file, **rename it to `credentials.json`**, and place it in the root folder of this project (`/Users/ayushpandey/Desktop/otml-rag/`).
7.  *Note: On the first run, a browser window will open asking you to login. This creates a `token.json` file automatically.*

## 4. Running the App

Run the Streamlit application:
```bash
streamlit run main2.py
```

## 5. Usage

1.  Open the URL provided by Streamlit (usually `http://localhost:8501`).
2.  **Upload**: Use the sidebar to upload a PDF.
3.  **Process**: Click "Process Document". Watch the status (Extracting -> Uploading -> Indexing).
4.  **Chat**: Ask questions in the main chat box.
5.  **Citations**: Expand the "References" below the answer to see which file and page the info came from.

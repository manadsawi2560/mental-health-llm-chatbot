import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings

load_dotenv()

DATA_DIR = os.path.join("data", "medical")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")

# ปิด telemetry และกำหนดให้ persistent ชัดเจน
CLIENT_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=CHROMA_DIR,
)

def load_docs(data_dir):
    docs = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            path = os.path.join(root, fn)
            if fn.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif fn.lower().endswith((".txt", ".md")):
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
    return docs

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)

    raw_docs = load_docs(DATA_DIR)
    if not raw_docs:
        print(f"[ingest] No documents found in {DATA_DIR}. Add PDFs or TXT/MD files and rerun.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ใช้ collection_name คงที่ป้องกันชน และส่ง client_settings เพื่อความเข้ากันได้ของ schema
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        client_settings=CLIENT_SETTINGS,
        collection_name="medibot",
    )
    vectordb.persist()
    print(f"[ingest] Indexed {len(splits)} chunks into {CHROMA_DIR} (collection: 'medibot').")

if __name__ == "__main__":
    main()

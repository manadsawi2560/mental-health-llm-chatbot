import os
from datetime import datetime
from typing import List
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from chromadb.config import Settings

# LLM backends
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
LANGUAGE_HINT = os.getenv("LANGUAGE_HINT", "th")

# ปิด telemetry และกำหนดให้ persistent ชัดเจน
CLIENT_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=CHROMA_DIR,
)

SAFETY_SYSTEM = (
    "คุณเป็นแชตบอทข้อมูลสุขภาพเพื่อการศึกษาเท่านั้น ไม่ใช่แพทย์ ไม่ให้การวินิจฉัย "
    "หรือสั่งการรักษา ห้ามให้คำแนะนำที่อาจเป็นอันตราย และควรแนะนำให้พบผู้เชี่ยวชาญเมื่อเหมาะสม "
    "ให้ตอบอย่างระมัดระวัง ใช้ภาษาง่าย ชี้แจงข้อจำกัด และระบุแหล่งที่มาจากบริบทที่ให้มาเมื่อเป็นไปได้."
)

def get_llm():
    if LLM_BACKEND == "ollama":
        return Ollama(model=OLLAMA_MODEL, temperature=0.2)
    elif LLM_BACKEND == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Check your .env")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=MAX_TOKENS)
    else:
        raise ValueError(f"Unknown LLM_BACKEND={LLM_BACKEND} (use 'ollama' or 'openai')")

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        client_settings=CLIENT_SETTINGS,
        collection_name="medibot",
    )
    return vectordb.as_retriever(search_kwargs={"k": 4})

def format_docs(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] {d.page_content}")
    return "\n\n".join(parts)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SAFETY_SYSTEM + " ตอบเป็นภาษาไทยโดยอัตโนมัติ เว้นแต่ผู้ใช้พิมพ์เป็นภาษาอังกฤษ."),
    ("human",
     "วันที่ปัจจุบัน: {today}\n"
     "คำถามผู้ใช้: {question}\n\n"
     "ใช้ข้อมูลใน 'บริบท' เพื่อช่วยตอบ หากข้อมูลไม่พอ ให้บอกอย่างซื่อสัตย์และแนะนำให้พบแพทย์.\n\n"
     "บริบท:\n{context}\n\n"
     "โปรดตอบอย่างสั้น กระชับ และปลอดภัย พร้อมคำเตือนที่เหมาะสม."
    ),
])

def answer(question: str) -> str:
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(question)
    context = format_docs(docs) if docs else "ไม่มีบริบทจากฐานข้อมูลในเครื่อง"
    llm = get_llm()
    prompt = PROMPT.format_prompt(today=datetime.now().strftime("%Y-%m-%d"), question=question, context=context)
    res = llm.invoke(prompt.to_string())

    # ครอบคลุมรูปแบบผลลัพธ์จาก .invoke()
    if isinstance(res, str):
        return res
    if hasattr(res, "content"):
        return res.content
    try:
        return str(res)
    except Exception:
        return "ไม่สามารถอ่านผลลัพธ์จากโมเดลได้ (ผลลัพธ์ไม่อยู่ในรูปแบบข้อความที่รองรับ)"

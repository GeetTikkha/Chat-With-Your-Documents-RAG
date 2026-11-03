import os
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

load_dotenv()
st.set_page_config(page_title="Chat With Your Documents", page_icon="📚")
st.title("📚 Chat With Your Documents")
st.caption("Upload documents and ask questions grounded in their content.")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in DocxDocument(file).paragraphs)
    if name.endswith(".txt"):
        return file.getvalue().decode("utf-8", errors="ignore")
    raise ValueError("Supported files: PDF, DOCX, TXT")

def chunk_text(text, size=700, overlap=120):
    words=text.split()
    chunks=[]
    start=0
    while start < len(words):
        end=min(start+size,len(words))
        chunks.append(" ".join(words[start:end]))
        if end==len(words): break
        start=end-overlap
    return chunks

def make_index(chunks):
    vectors=embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    vectors=np.asarray(vectors,dtype="float32")
    idx=faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    return idx

def search(question,chunks,index,k):
    q=embedder.encode([question],convert_to_numpy=True,normalize_embeddings=True)
    scores,ids=index.search(np.asarray(q,dtype="float32"),min(k,len(chunks)))
    return [(float(s),chunks[int(i)]) for s,i in zip(scores[0],ids[0]) if i>=0]

api_key=os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("Add OPENAI_API_KEY to your .env file or hosting secrets.")
    st.code("OPENAI_API_KEY=your_key_here\nOPENAI_MODEL=gpt-4o-mini")
    st.stop()
client=OpenAI(api_key=api_key)
model=os.getenv("OPENAI_MODEL","gpt-4o-mini")

with st.sidebar:
    st.header("Retrieval settings")
    chunk_size=st.slider("Chunk size (words)",300,1200,700,100)
    overlap=st.slider("Overlap (words)",0,250,120,20)
    top_k=st.slider("Retrieved chunks",1,8,4)

files=st.file_uploader("Upload PDF, DOCX or TXT",type=["pdf","docx","txt"],accept_multiple_files=True)
if files:
    chunks=[]
    for f in files:
        try:
            extracted=extract_text(f)
            chunks.extend([f"[Source: {f.name}] {c}" for c in chunk_text(extracted,chunk_size,min(overlap,chunk_size-1))])
        except Exception as e:
            st.error(f"Could not process {f.name}: {type(e).__name__}")
    if not chunks:
        st.warning("No readable text found. Scanned PDFs may require OCR.")
        st.stop()
    with st.spinner("Creating embeddings and FAISS index..."):
        index=make_index(chunks)
    st.success(f"Indexed {len(chunks)} chunks from {len(files)} document(s).")
    if "history" not in st.session_state: st.session_state.history=[]
    for m in st.session_state.history:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    question=st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.history.append({"role":"user","content":question})
        with st.chat_message("user"): st.markdown(question)
        found=search(question,chunks,index,top_k)
        context="\n\n".join(f"Passage {i+1} (similarity {s:.3f}):\n{t}" for i,(s,t) in enumerate(found))
        try:
            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    r=client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role":"system","content":"Answer only from the supplied document context. If the answer is absent, say you cannot find it. Do not invent facts."},
                            {"role":"user","content":f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
                        ],
                        temperature=0.2,max_tokens=500
                    )
                    answer=r.choices[0].message.content or ""
                    st.markdown(answer)
                    with st.expander("View retrieved passages"):
                        for i,(score,passage) in enumerate(found,1):
                            st.markdown(f"**Passage {i} · similarity {score:.3f}**")
                            st.write(passage)
            st.session_state.history.append({"role":"assistant","content":answer})
        except Exception as e:
            st.error(f"API request failed: {type(e).__name__}. Check key, model access and quota.")
else:
    st.info("Upload at least one document to begin.")

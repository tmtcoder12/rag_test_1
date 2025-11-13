# conversational_rag.py
# Fully local conversational RAG: EmbeddingGemma (embeddings) + FAISS (retrieval) + Llama/Gemma (generation) via Ollama
import requests, numpy as np, faiss, textwrap, datetime, json, sys

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma:latest"
LLM_MODEL   = "gemma3:4b"        # or "llama3.1:8b" etc.
NUM_CTX     = 4096                # adjust to your model
TOP_K       = 5                   # retrieved passages per turn

# ---------- Load corpus ----------
with open('formatted_data.json', 'r', encoding='utf-8') as f:
    DATA = json.load(f)
TEXTS = [d["text"] for d in DATA]

# ---------- Embeddings ----------
def embed_texts(texts, model=EMBED_MODEL):
    vecs = []
    for t in texts:
        r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                          json={"model": model, "prompt": t},
                          timeout=120)
        r.raise_for_status()
        emb = r.json().get("embedding")
        if emb is None:
            raise RuntimeError(f"No 'embedding' returned for: {t[:80]}")
        vecs.append(emb)
    X = np.array(vecs, dtype="float32")
    # Normalize for inner-product cosine
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

# ---------- Build FAISS index once ----------
X = embed_texts(TEXTS)
DIM = X.shape[1]
INDEX = faiss.IndexFlatIP(DIM)
INDEX.add(X)
ID2DOC = {i: DATA[i] for i in range(len(DATA))}

def retrieve(query, k=TOP_K):
    qv = embed_texts([query])[0].reshape(1, -1)
    D, I = INDEX.search(qv, k)
    out = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        d = ID2DOC[int(idx)]
        out.append({"score": float(score), **d})
    return out

# ---------- Prompt builder (per turn) ----------
SYSTEM_MSG = (
    "You are a careful clinical assistant for a family clinic.\n"
    "Use ONLY the supplied Context for answers. If missing, say you don't know.\n"
    "Cite sources inline like [id · patient_id]. Keep answers concise and safe."
)

def build_turn_content(user_query, ctx_docs, max_snip_len=800):
    context = "\n\n---\n".join(
        f"[{d['id']} · {d['patient_id']}]\n" + textwrap.shorten(d["text"], width=max_snip_len)
        for d in ctx_docs
    )
    return (
        "Context:\n" + context + "\n\n"
        "Question: " + user_query + "\n"
        "Answer:"
    )

# ---------- Ollama chat wrapper ----------
def ollama_chat(messages, temperature=0.0, num_ctx=NUM_CTX):
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    """
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
            "stream": False
        },
        timeout=180
    )
    r.raise_for_status()
    data = r.json()
    # Ollama /api/chat returns {"message": {"role": "assistant", "content": "..."} , ...}
    return data.get("message", {}).get("content", "").strip()

# ---------- Chat session manager ----------
class ConversationalRAG:
    def __init__(self, system_msg=SYSTEM_MSG):
        self.history = [{"role":"system", "content": system_msg}]

    def ask(self, user_query):
        # 1) retrieve fresh context for THIS turn
        hits = retrieve(user_query, k=TOP_K)
        turn_content = build_turn_content(user_query, hits)

        # 2) Add ONLY the user question (not the long context) to history to keep it clean
        self.history.append({"role":"user", "content": user_query})

        # 3) Send a composite user message for this turn that includes the context + question
        #    but do NOT store that composite in history permanently (avoids ballooning)
        tmp_messages = self.history[:-1] + [
            {"role":"user", "content": turn_content}
        ]

        # 4) Chat
        answer = ollama_chat(tmp_messages)

        # 5) Append just the final answer to history
        self.history.append({"role":"assistant", "content": answer})

        # 6) Return answer plus the retrieved doc ids for transparency
        citations = [(d["id"], d["patient_id"], d["score"]) for d in hits]
        return answer, citations

    def reset(self):
        self.history = [{"role":"system", "content": SYSTEM_MSG}]

# ---------- Simple REPL ----------
def run_chat():
    chat = ConversationalRAG()
    print("Conversational RAG ready. Type /reset or /exit.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in ("/exit", "/quit"):
            print("Bye!")
            break
        if q.lower() == "/reset":
            chat.reset()
            print("(history cleared)")
            continue

        answer, cites = chat.ask(q)
        print("\nAssistant:", answer)
        if cites:
            print("\nSources:")
            for (doc_id, patient_id, score) in cites:
                print(f"- [{doc_id} · {patient_id}]  (score={score:.3f})")

if __name__ == "__main__":
    run_chat()

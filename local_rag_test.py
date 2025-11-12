# Fully local RAG: EmbeddingGemma (embeddings) + FAISS (retrieval) + Llama3.1 (generation) via Ollama
import requests, numpy as np, faiss, textwrap, datetime
import json

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma:latest"
LLM_MODEL   = "gemma3:4b"   # change to any local generator you have

# Open and load data the JSON file 
with open('formatted_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# --- 2) Helpers: call Ollama embeddings & generation ---
def embed_texts(texts, model=EMBED_MODEL, batch_size=16):
    """Return a numpy array of shape (N, D) for a list of strings."""
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        print(f"batch{i} is being processed")
        # Ollama embeddings currently expects a single string; call per item for simplicity
        for t in batch:

            print(f"This is t: {t}")
            
            r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                              json={"model": model, "prompt": t}, timeout=120)
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if emb is None:
                raise RuntimeError(f"No 'embedding' in response for text: {t[:80]}...")
            vecs.append(emb)
    X = np.array(vecs, dtype="float32")
    # Normalize for cosine similarity via inner product
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def ollama_generate(prompt, model=LLM_MODEL, temperature=0.0, num_ctx=2048):
    """Use /api/generate for simple single-turn generation."""
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": model,
                            "prompt": prompt,
                            "options": {"temperature": temperature, "num_ctx": num_ctx},
                            "stream": False},
                      timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()



# --- 3) Build embeddings + FAISS index ---
texts = [d["text"] for d in data]
X = embed_texts(texts)                # (N, D), normalized
dim = X.shape[1]
index = faiss.IndexFlatIP(dim)        # inner product == cosine for normalized vectors
index.add(X)
id2doc = {i: data[i] for i in range(len(data))}



# --- 4) Retrieval ---
def retrieve(query, k=3):
    qv = embed_texts([query])[0].reshape(1, -1)  # single normalized query vector
    D, I = index.search(qv, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: 
            continue
        d = id2doc[int(idx)]
        results.append({"score": float(score), **d})
    return results

# --- 5) Build RAG prompt ---
def build_prompt(query, ctx_docs, max_snip_len=800):
    context = "\n\n---\n".join(
        f"[{d['id']} · {d['patient_id']}]\n" + textwrap.shorten(d["text"], width=max_snip_len)
        for d in ctx_docs
    )
    return f"""You are a careful assistant that works for a family clinic, and help up with minor diagnosises and information look up. Use ONLY the Context. If the answer is missing, say you don't know.
Cite sources inline like [id · patient_id].

Context:
{context}

Question: {query}
Answer:"""

#function that helps export queries and output into txt. file
def run_batch_and_save(queries, k=5, out_prefix="batch_results"):
    # Create a timestamped output file name
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{out_prefix}_{ts}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, query in enumerate(queries, start=1):
            # --- retrieval + LLM ---
            hits = retrieve(query, k=k)
            prompt = build_prompt(query, hits)
            answer = ollama_generate(prompt)

            # --- console output (optional) ---
            print(f"\n=== Query {idx} ===")
            print(f"Query: {query}")
            print("Top hits:")
            for h in hits:
                print(f"- {h['id']} :: {h['patient_id']}  (score={h['score']:.3f})")
            print("\n--- Prompt sent to LLM ---\n")
            print(prompt)
            print("\n--- LLM Answer ---\n")
            print(answer)

            # --- write to file ---
            f.write(f"=== Query {idx} ===\n")
            f.write(f"Query: {query}\n\n")

            f.write("Top hits:\n")
            for h in hits:
                f.write(f"- {h['id']} :: {h['patient_id']}  (score={h['score']:.3f})\n")

            f.write("\n--- Prompt sent to LLM ---\n")
            f.write(prompt + "\n\n")

            f.write("--- LLM Answer ---\n")
            f.write(str(answer) + "\n")

            f.write("\n" + ("=" * 60) + "\n\n")

    print(f"\nAll results saved to: {out_path}")
    return out_path


#Demo run
if __name__ == "__main__":
    # Put your queries here
    queries = [
    # Direct lookups (factual)
    "Does Emily Chen have any medication allergies?",
    "What inhalers does Emily Chen currently use?",
    "On 2023-12-18, why did Emily Chen visit the clinic?",
    "What dose of lisinopril is David Morales prescribed initially?",
    "What was David Morales’s LDL result at his 2023-12-05 visit?",
    "What antibiotic was prescribed for Sofia Rodriguez’s sore throat on 2023-02-15?",
    "What is Michael Turner’s documented allergy?",
    "What medication does Aisha Patel take for anxiety, and at what dose?",
    "Which patient has a shellfish allergy and what type of allergy?",
    "Which patient is on metformin and why?",

    # Paraphrase / synonyms
    "Did Emily report breathing issues during allergy season?",
    "Has David’s blood pressure management improved by 2025?",
    "Was Sofia’s January 2025 cough likely bacterial or post-viral?",
    "Did Michael receive guidance about weight management in late 2024?",
    "Did Hannah’s headache frequency decrease by April 2024?",

    # Date-bounded queries
    "What were Emily Chen’s plans at the 2024-03-25 check-up?",
    "What changes were made to David’s statin therapy on 2023-12-05?",
    "What was the assessment for Sofia’s 2024-01-20 ear pain visit?",
    "What was Michael’s A1C and assessment on 2024-03-02?",
    "What interventions were recorded for Hannah on 2025-01-08?",

    # Multi-hop / aggregation
    "List all vaccines administered across these records by patient and date.",
    "For Marcus Brown, provide me the progression of his back pain.",
    "Which patients received antibiotics, what for, and when.",
    "Identify all documented lifestyle recommendations (diet, exercise, sleep) by patient.",

    # Negation / contrast
    "Which patients explicitly have no documented allergies?",
    "Did Sofia ever present with fever at the 2025-01-03 visit?",
    "Was Hannah ever documented to have visual aura with migraines?",

    # Entity & value extraction
    "Extract all medication names and doses for Aisha across her timeline.",
    "Pull every lab value mentioned (LDL, A1C, glucose) with dates and patients.",
    "List all family-history conditions per patient.",

    # “Find the best source chunk”
    "Where (which note) is “exercise-induced bronchospasm” mentioned, and what plan follows it?",
    "Find the note that contains “HEPA air filter”; which patient and visit is it?",
    "Locate the record with “digital eye strain”; what was the plan?",

    # Safety / edge cases
    "Which patient is prescribed insulin?",
    "Do any records mention hospitalization or ER admission?",
    "Are there any documented medication side effects besides allergies?"
]

    run_batch_and_save(queries, k=5)

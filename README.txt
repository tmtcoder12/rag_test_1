The goal of this first test: 
to measure the performance of a small RAG system with a small sythetic formatted dataset. 


How is performace tracked and what is it in this context: 
I will tracked performace by determining whether query answers are relevant or not-relevant to the query, and dataset used for the RAG system. 

What is a "small synthetic formatted dataset":
Small in the context that the data set contains around 2800 words (all text). This would be around 17 ish KB. 
On average, English words are about 5 characters long, plus one space or punctuation → roughly 6 bytes per word in plain UTF-8 text.
2,800 words × 6 bytes ≈ 16,800 bytes ≈ 16.8 KB (calculation was provided by ChatGPT). I say formatted because the data is in JSON format, and 
may RAG systems don't involve structured data.

Hardware and RAG Architechture: 

Hardware: 
Using an Apple M4 studio (36 gb unified ram). 

Architechture: 
Embedding Model: embeddinggemma:300m
Vector DB: FAISS
Retrieval Algorithm: IndexFlatIP (inner product, cosine for normalized vectors), via FAISS library 
LLM: gemma3:4b


Questions that I will be asking:

- How many visits did personX make in 2023? 
- PersonX is coming in for a visit with X, Y, Z symptoms, have they had similar symptoms in the past? 
- What medicaitons is personX taking? 
- Based on all of personX's visit data, what could be potential future health complications?






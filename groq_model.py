# groq_model.py
import numpy as np
from groq import Groq

class GroqModel:
    def __init__(self, embedder, index, all_chunks, pdf_text_full, csv_text_full, groq_api_key):
        """
        Initialize the GroqModel with precomputed components.
        """
        self.embedder = embedder
        self.index = index
        self.all_chunks = all_chunks
        self.pdf_text_full = pdf_text_full
        self.csv_text_full = csv_text_full
        self.client = Groq(api_key=groq_api_key)

    def ask_groq(self, question, mode="rag", top_k=4):
        """
        Use the Groq API to answer a question.
          - "rag": Retrieves relevant chunks using the FAISS index.
          - "all_docs": Uses full document text as context.
          - Others: Sends the question directly.
        """
        if mode == "rag":
            try:
                query_embed = self.embedder.encode([question])
                _, indices = self.index.search(np.array(query_embed).astype("float32"), top_k)
                context = "\n".join([self.all_chunks[i] for i in indices[0]])
                prompt = (
                    f"Berdasarkan informasi berikut (jawablah dalam Bahasa Indonesia)!\n{context}\n\n"
                    f"Pertanyaan: {question}\nJawaban:"
                )
            except Exception as e:
                return f"Error in RAG mode retrieval: {e}"
        elif mode == "all_docs":
            try:
                context = self.pdf_text_full + " " + self.csv_text_full
                prompt = (
                    f"Berdasarkan informasi berikut (jawablah dalam Bahasa Indonesia)!\n{context}\n\n"
                    f"Pertanyaan: {question}\nJawaban:"
                )
            except Exception as e:
                return f"Error in ALL_DOCS mode processing: {e}"
        else:
            prompt = f"Jawablah dalam Bahasa Indonesia!\n{question}"

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",  # Update as needed
                temperature=0.9,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in LLM API call: {e}"

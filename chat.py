# chat.py
import os
from dotenv import load_dotenv
from input_process import DataProcessor
from embedding_vector import EmbeddingVector
from groq_model import GroqModel

load_dotenv()  # Load environment variables from .env

def main():
    csv_path = "knowledge_resource/csv_resource.csv"
    pdf_path = "knowledge_resource/pdf_resource.pdf"

    # Process input data using DataProcessor
    dp = DataProcessor()
    csv_chunks = dp.process_csv(csv_path)
    pdf_text = dp.process_pdf(pdf_path)
    pdf_chunks = dp.chunk_pdf_text(pdf_text)
    csv_chunks_combined = dp.chunk_csv_text(csv_chunks)

    all_chunks = pdf_chunks + csv_chunks_combined

    # Build embeddings and FAISS index using EmbeddingVector
    ev = EmbeddingVector()
    embeddings = ev.build_embeddings(all_chunks)
    index = ev.build_faiss_index(embeddings)
    csv_text_full = " ".join(csv_chunks)
    pdf_text_full = pdf_text

    # Retrieve GROQ API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("Error: GROQ_API_KEY not set in environment variables.")
        return

    model = GroqModel(ev.embedder, index, all_chunks, pdf_text_full, csv_text_full, groq_api_key)

    # Define a list of 10 questions
    questions_list = [
        "Siapa itu Vin Corp? dan bagaimana Vinc Corp melakukan pengambilan keputasan bisnis agar tetap relevan?",
        "Apa kegunaan Vnelia VOC? dan jelaskan kenapa saya harus menggunakannya",
        "Jelaskan kandungan teknis didalam produk Vnelia VOC?",
        "Bagaimana produk Vnelia VOC dikemas dan didistribusikan?",
        "Bagaimana saya dapat menghubungi layanan pelanggan?",
        "Jelaskan tren produk terjual pada tahun 2023 - 2024?",
        "Bagaimana umpan balik atau review dari pelanggan mengenai Vnelia VOC?",
        "Berapa total transaksi dan volume item yang diproses pada 2024-03-01?",
        "Dimanakah lokasi pelanggan pada transaksi dengan code TRX001?",
        "Apa metode pembayaran yang paling sering digunakan/populer dalam dataset transaksi?"   
    ]

    # Iterate over each question and generate the answer using the GroqModel's ask_groq method.
    for i, question in enumerate(questions_list, start=1):
        print("\n" + "="*100)
        print(f"Question {i}: {question}")

        # You can change the mode ("rag", "all_docs", "base_model") as needed.
        answer_rag = model.ask_groq(question, mode="rag")
        answer_all_docs = model.ask_groq(question, mode="all_docs")
        answer_base_model = model.ask_groq(question, mode="base_model")

        print("\n*Answer RAG*:", answer_rag)
        print("\n*Answer ALL_DOCS*:", answer_all_docs)
        print("\n*Answer BASE_MODEL*:", answer_base_model)
        print("="*100 + "\n")

if __name__ == "__main__":
    main()

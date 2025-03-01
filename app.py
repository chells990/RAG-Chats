# app.py
import os
import gradio as gr
from dotenv import load_dotenv
from input_process import DataProcessor
from embedding_vector import EmbeddingVector
from groq_model import GroqModel

# Load environment variables
load_dotenv()

# Initialize components once at startup
def initialize_system():
    # Process data
    dp = DataProcessor()
    csv_chunks = dp.process_csv("knowledge_resource/csv_resource.csv")
    pdf_text = dp.process_pdf("knowledge_resource/pdf_resource.pdf")
    pdf_chunks = dp.chunk_pdf_text(pdf_text)
    csv_chunks_combined = dp.chunk_csv_text(csv_chunks)
    all_chunks = pdf_chunks + csv_chunks_combined

    # Load precomputed embeddings
    ev = EmbeddingVector()
    embeddings = ev.build_embeddings(all_chunks) 
    index = ev.build_faiss_index(embeddings)  # Build FAISS index
    
    # Initialize model
    return GroqModel(
        ev.embedder,
        index,
        all_chunks,
        pdf_text,
        " ".join(csv_chunks),
        os.getenv('GROQ_API_KEY')
    )

# Initialize system
model = initialize_system()

# Chat interface
def respond(question, mode):
    return model.ask_groq(question, mode=mode)

# Gradio UI
with gr.Blocks(title="Vnelia Chatbot", theme=gr.themes.Soft()) as demo:
    # Header
    gr.Markdown("""
    # 🇮🇩 Vnelia VOC Chatbot  
    Selamat datang di chatbot Vnelia VOC! Ajukan pertanyaan Anda di bawah ini.
    """)

    # Input Section
    with gr.Row():
        question = gr.Textbox(
            label="Pertanyaan Anda",
            placeholder="Contoh: Apa itu Vnelia VOC?",
            lines=2
        )
        mode = gr.Dropdown(
            choices=["rag", "all_docs", "base_model"],
            value="rag",
            label="Mode Jawaban",
            info="Pilih mode untuk menghasilkan jawaban."
        )
    
    # Output Section
    answer = gr.Textbox(
        label="Jawaban",
        placeholder="Jawaban akan muncul di sini...",
        lines=5,
        interactive=False
    )

    # Submit Button
    submit = gr.Button("Ajukan Pertanyaan", variant="primary")

    # Interaction
    submit.click(
        fn=respond,
        inputs=[question, mode],
        outputs=answer
    )

    # Footer
    gr.Markdown("""
    ---
    **Mode Penjelasan**:
    - **RAG**: Menggunakan Retrieval-Augmented Generation untuk jawaban yang lebih akurat.
    - **All Docs**: Menggunakan seluruh dokumen sebagai konteks.
    - **Base Model**: Menggunakan model dasar tanpa konteks tambahan.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
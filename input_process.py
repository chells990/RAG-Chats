# input_process.py
import pandas as pd
import re
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataProcessor:
    def process_csv(self, file_path):
        """
        Process a CSV file and return a list of text chunks that describe transactions.
        """
        df = pd.read_csv(file_path)
        text_chunks = []

        # Build transaction descriptions
        for _, row in df.iterrows():
            transaction_desc = (
                f"Transaksi {row['transaction_id']} tercatat saat {row['time_stamp']} "
                f"dengan status {row['status']}. "
                f"Transaksi dilakukan melalui {row['channel']} "
                f"dengan metode pembayaran {row['payment_method']}. "
                f"Pelanggan berada di {row['cust_location']} "
                f"dan membeli {row['quantity']} unit {row['item_type']} "
                f"dengan harga {row['price_per_unit']} per unit. "
                f"Total pendapatan transaksi ini adalah {row['total_revenue']}."
            )
            text_chunks.append(transaction_desc)

        # Build daily summaries
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        df['date'] = df['time_stamp'].dt.date
        daily_summaries = df.groupby('date').agg({
            'transaction_id': 'count',
            'quantity': 'sum'
        }).reset_index()

        for _, row in daily_summaries.iterrows():
            summary_desc = (
                f"Pada tanggal {row['date']}, terdapat {row['transaction_id']} transaksi. "
                f"Total volume adalah {row['quantity']} unit."
            )
            text_chunks.append(summary_desc)

        # Payment method statistics
        payment_stats = df['payment_method'].value_counts().reset_index()
        payment_stats.columns = ['payment_method', 'count']
        for _, row in payment_stats.iterrows():
            stats_desc = (
                f"Metode pembayaran {row['payment_method']} digunakan sebanyak {row['count']} kali."
            )
            text_chunks.append(stats_desc)

        # Customer location statistics
        location_stats = df['cust_location'].value_counts().reset_index()
        location_stats.columns = ['cust_location', 'count']
        for _, row in location_stats.iterrows():
            location_desc = (
                f"Lokasi pelanggan {row['cust_location']} memiliki {row['count']} transaksi."
            )
            text_chunks.append(location_desc)

        return text_chunks

    def process_pdf(self, file_path):
        """
        Extract and clean text from a PDF file.
        """
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            cleaned_text = re.sub(r'\s*\n\s*', ' ', page_text).strip()
            text += cleaned_text + " "
        return text

    def chunk_pdf_text(self, text, chunk_size=512, chunk_overlap=128):
        """
        Split pdf text into smaller chunks using RecursiveCharacterTextSplitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
            length_function=len,
            keep_separator=True
        )
        return splitter.split_text(text)

    def chunk_csv_text(self, csv_text_chunks, group_size=2):
        """
        Group CSV text chunks into combined strings.
        """
        grouped = [csv_text_chunks[i:i+group_size] for i in range(0, len(csv_text_chunks), group_size)]
        return [" ; ".join(chunk) for chunk in grouped]

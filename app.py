import fitz  # PyMuPDF
from io import BytesIO
import re
import os
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

CHROMA_BASE_PATH = "./db"
GROQ_API_KEY = 'Secret_key'
PDF_PATH = ".pdf"

# Update embedding model if a newer or more efficient model is available
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Check for newer models
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

model = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY
)


template = """
You are a professional cardiologist with expertise in cardiovascular medicine. Use your knowledge to provide medically accurate, clear, and concise responses based strictly on the context provided below.

Context: {context}

Patient Question: {question}

Guidelines:
- Stick to the information provided in the context, without speculating beyond it.
- Use medical terminology where appropriate, but ensure your answer is understandable to a non-specialist.
- If the context does not contain relevant information, respond with: "The context provided does not contain enough information to answer this question."
- Do not provide personal medical advice; offer general explanations where relevant.

Answer:
"""

prompt = PromptTemplate.from_template(template)

def filter_text(text: str) -> str:
    irrelevant_patterns = [
        r'^\d{1,3}$',  # Page numbers
        r'\b(Page|Reference|Copyright|MD Consult|User Name|Password|Member Log On)\b',  # Headers/footers
        r'\b(Ann Surg|Ann Vasc Surg|Am J Cardiol|Circulation|Angiology|N Engl J Med|JAMA)\b',  # Journals
        r'\d{4}',  # Years
        r'^\s*$',  # Empty lines
        r'^\d+\.',  # Numbered lists
        r'\b(LDL|HDL|TG|mg/dL|mmHg)\b',  # Medical abbreviations with units
        r'^\s*\d+[\.\)]',  # Numbered lines
        r'\bFig(ure)?\b'  # Figures (optional based on needs)
    ]

    combined_pattern = '|'.join(irrelevant_patterns)
    filtered_lines = [line for line in text.split('\n') if not re.search(combined_pattern, line, re.IGNORECASE)]
    filtered_text = "\n".join(filtered_lines)
    filtered_text = re.sub(r'\n+', '\n', filtered_text).strip()  # Remove excessive newlines
    filtered_text = re.sub(r'^\s+', '', filtered_text)  # Remove leading whitespace

    return filtered_text


import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_documents(pdf_stream: BytesIO):
    documents = []
    try:
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text = page.get_text("text")
            if text.strip():
                filtered_text = filter_text(text)
                if filtered_text.strip():
                    metadata = {"page_number": page_number + 1}
                    documents.append(Document(page_content=filtered_text, metadata=metadata))
        logging.info("Documents loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading documents: {e}")
    return documents

def query(question: str, top_k: int = 5) -> None:
    try:
        chroma_path = os.path.join(CHROMA_BASE_PATH)
        if not os.path.exists(chroma_path):
            logging.error("Chroma database does not exist.")
            return "Chroma database is missing."

        db = Chroma(persist_directory=chroma_path, embedding_function=embedding)
        results = db.similarity_search_with_relevance_scores(question, k=top_k)

        if not results or results[0][1] < 0.3:
            logging.info("No relevant documents found.")
            return "No relevant information found."

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        chain = prompt | model | StrOutputParser()
        response = chain.invoke({
            "context": context_text,
            "question": question
        })
        return response
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return f"An error occurred: {str(e)}"

def process_in_batches(pdf_stream: BytesIO, batch_size=50):
    pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
    total_pages = pdf_document.page_count

    for start_page in range(0, total_pages, batch_size):
        end_page = min(start_page + batch_size, total_pages)
        documents = []
        for page_number in range(start_page, end_page):
            page = pdf_document[page_number]
            text = page.get_text("text")
            filtered_text = filter_text(text)
            if filtered_text.strip():
                metadata = {"page_number": page_number + 1}
                documents.append(Document(page_content=filtered_text, metadata=metadata))

        if documents:
            chunks = split_text(documents)
            save_to_chroma(chunks)
        # Clear memory after processing each batch
        del documents
        torch.cuda.empty_cache()  # Free up GPU memory

    print("PDF processed in batches.")

def split_text(documents: list[Document]):
    # Adjust chunk_size and overlap based on document length and complexity
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def save_to_chroma(chunks: list[Document]):
    chroma_path = os.path.join(CHROMA_BASE_PATH)
    try:
        if os.path.exists(chroma_path):
            db = Chroma(persist_directory=chroma_path, embedding_function=embedding)
            db.add_documents(chunks)
            logging.info("Chroma database updated successfully.")
        else:
            logging.info(f"Creating new Chroma database in directory {chroma_path}")
            db = Chroma.from_documents(chunks, embedding, persist_directory=chroma_path)
            logging.info("Chroma database created successfully.")
    except Exception as e:
        logging.error(f"Error saving to Chroma: {e}")
    finally:
        del chunks
        torch.cuda.empty_cache()

with open(PDF_PATH, 'rb') as f:
    pdf_stream = BytesIO(f.read())
process_in_batches(pdf_stream, batch_size=100)

import gradio as gr

async def chatbot_response(user_question: str) -> str:
    try:
        response = query(user_question)
        return response
    except Exception as e:
        logging.error(f"Error in chatbot response: {e}")
        return f"An error occurred: {str(e)}"

custom_css = """
/* Custom styling for the Gradio interface */
.input-textbox, .output-textbox {
    font-size: 16px;
    padding: 10px;
    width: 100%;
}
.submit-button {
    background-color: #FF6347;
    color: white;
    font-size: 16px;
    padding: 10px;
    border-radius: 5px;
    cursor: pointer;
}
"""

with gr.Blocks(css=custom_css) as interface:
    gr.Markdown("<h1>Braunwald Cardiovascular Medicine Chatbot</h1>")
    gr.Markdown("<h3>Ask any question related to cardiovascular medicine.</h3>")

    user_input = gr.Textbox(lines=2, placeholder="Ask a medical question...", label="Enter your question", elem_classes="input-textbox")
    chatbot_output = gr.Textbox(label="Chatbot Response", elem_classes="output-textbox")

    submit_btn = gr.Button("Submit", elem_classes="submit-button")
    submit_btn.click(chatbot_response, inputs=user_input, outputs=chatbot_output, show_progress=True)  # Show loading indicator

interface.launch()

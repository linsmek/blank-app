#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:31:18 2024

@author: linamekouar
"""

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings  # Local embeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Path to the directory containing multiple PDFs
DATA_PATH = "/Users/linamekouar/Desktop/SYLLABI"

# Loading all PDF documents from the directory
def load_documents(data_path):
    """Load all PDFs from the directory and return a list of document objects."""
    documents = []
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(data_path, file)
            print(f"Processing {pdf_path}...")
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents

# Splitting documents into chunks
def split_documents(documents):
    """Split the documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust chunk size based on your needs
        chunk_overlap=100,  
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

# Embedding the chunks
def create_vector_store(chunks):
    """Embed the chunks and store them in FAISS."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    chunk_texts = [chunk.page_content for chunk in chunks]
    vector_store = FAISS.from_texts(chunk_texts, embeddings)
    print("Vector store created and populated.")
    return vector_store

# Main Workflow
if __name__ == "__main__":
    # Load documents
    documents = load_documents(DATA_PATH)
    if not documents:
        print("No documents found. Exiting.")
        exit()
    print(f"Loaded {len(documents)} documents.")

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    llm = Ollama(model="llama3.2")

    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Query the system
    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = qa_chain.run(query)
        print("Answer:", response)

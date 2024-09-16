# Braunwald_Cardiovascular_Chatbot
This repository contains the code for a cardiovascular medicine chatbot built using LangChain, HuggingFace, and Gradio. The chatbot is designed to answer questions related to cardiovascular medicine based on the text extracted from the "Braunwald Heart Disease: A Textbook of Cardiovascular Medicine" PDF. 
It uses advanced NLP models and a vector database for document retrieval and similarity searches. The application leverages GPU acceleration to enhance performance during embeddings and model inference.
The project aims to assist medical professionals and students by providing contextually accurate responses to questions about cardiovascular medicine.
Following are its Key Features:-
  1.  PDF Processing - Here I have used Braunwald Heart Disease, a renowned book among Cardiovascular Specialists.
  2.  GPU Utilization - Will accelerate embeddings and model inference for faster performance.
  3.  Vector Database - Chroma for storing and retrieving document embeddings for context-based Q/ A.
  4.  LangChain and HuggingFace - Uses Huggingface embeddings and Langchain pipeline.
  5.  Gradio Interface - A user friendly interface to interact with user.
  6.  Batch wise Processing - Handles large PDF docs in batches to optimize memory and processing time.

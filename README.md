Jupiter FAQ Chatbot (Streamlit)
Overview
This project develops a human-friendly FAQ chatbot for Jupiter's services. It scrapes existing FAQ data from the Jupiter Money website, preprocesses it, and then uses semantic search to provide intelligent, conversational, and accurate answers to user queries. The application is built using Streamlit, providing an interactive web interface.

Features
Web Scraping: Automatically extracts FAQ questions and answers from the Jupiter Money contact/help page.

Data Preprocessing: Cleans and normalizes scraped text, removes duplicate questions, and categorizes FAQs into topics like KYC, Rewards, Payments, and Limits.

Semantic Search: Utilizes SentenceTransformer embeddings (all-MiniLM-L6-v2) and FAISS (Facebook AI Similarity Search) for efficient and accurate retrieval of the most semantically similar FAQ answer to a user's query.

Confidence Handling: Implements a threshold-based mechanism to determine if a relevant answer is found, responding gracefully if unsure.

Conversational Interface: Provides a user-friendly chat experience using Streamlit's chat elements, maintaining conversation history.

Model Caching: Leverages Streamlit's @st.cache_resource to optimize performance by loading models and processing data only once.

(Optional) Answer Paraphrasing: Includes functionality using prithivida/parrot_paraphraser_on_T5 to rephrase answers into more natural language (currently commented out but integrated).

How it Works
Scraping: The application fetches the HTML content from https://jupiter.money/contact/ and extracts question-answer pairs from the designated FAQ section.

Cleaning & Structuring: Extracted data is cleaned (e.g., whitespace removal, special character stripping) and stored in a Pandas DataFrame. Duplicate questions are identified and removed based on their cleaned text.

Categorization: Questions are assigned categories based on keywords.

Embedding: Each FAQ question is converted into a high-dimensional vector (embedding) using the all-MiniLM-L6-v2 Sentence Transformer model.

Indexing: These embeddings are added to a FAISS index, enabling fast nearest-neighbor search.

User Interaction:

A user types a question into the Streamlit chat interface.

The user's question is embedded using the same Sentence Transformer model.

The embedding is used to query the FAISS index to find the most similar FAQ question.

If the similarity score (distance) is below a predefined threshold, the corresponding answer is retrieved and displayed.

If no sufficiently similar match is found, the bot indicates that it doesn't understand the query.

The conversation history is maintained using Streamlit's session state.

Setup and Installation
To run this application locally, follow these steps:

Clone the repository (or save the code):
If you have the code as a single file, save it as chatbot_app.py (or any .py extension).

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:
Install all required Python packages using pip.

pip install streamlit requests beautifulsoup4 pandas transformers sentence-transformers faiss-cpu torch numpy

Running the Application
Once the dependencies are installed, you can run the Streamlit application from your terminal:

streamlit run chatbot_app.py

This command will open the Streamlit application in your default web browser.

Project Structure
The core logic is contained within a single Python script (chatbot_app.py in this example).

load_and_process_faq_data(): Handles web scraping, data cleaning, deduplication, and categorization.

load_models_and_create_index(): Loads the prithivida/parrot_paraphraser_on_T5 (for paraphrasing) and all-MiniLM-L6-v2 (for embeddings) models, and sets up the FAISS index.

paraphrase_answer(): (Optional) Function to rephrase answers using the T5-based model.

find_best_faq_match(): Performs the semantic search against the FAISS index and applies the confidence threshold.

Streamlit UI logic: Manages the web interface, user input, chat display, and session state.

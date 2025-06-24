🚀 Jupiter FAQ Chatbot (Streamlit) Overview
This project develops a human-friendly FAQ chatbot for Jupiter's services. It scrapes FAQ data from the Jupiter Money website, preprocesses it, and uses semantic search to provide intelligent, conversational, and accurate answers. The application is built using Streamlit, providing an interactive web interface.

🌟 Features
Web Scraping: Automatically extracts FAQ questions and answers from the Jupiter Money contact/help page.

Data Preprocessing: Cleans and normalizes scraped text, removes duplicate questions, and categorizes FAQs into topics like KYC, Rewards, Payments, and Limits.

Semantic Search: Utilizes SentenceTransformer embeddings (all-MiniLM-L6-v2) and FAISS (Facebook AI Similarity Search) for fast, accurate semantic matching.

Confidence Handling: Uses a threshold-based mechanism to determine if a relevant answer exists and responds gracefully when unsure.

Conversational Interface: Streamlit's chat interface maintains a natural user experience and retains conversation history.

Model Caching: Implements @st.cache_resource to optimize performance by loading models and processing data only once.

(Optional) Answer Paraphrasing: Includes a T5-based model (prithivida/parrot_paraphraser_on_T5) to rephrase answers (currently commented out but integrated).

⚙️ How it Works
🔹 Scraping
Fetches HTML content from https://jupiter.money/contact/

Extracts question-answer pairs from the FAQ section

🔹 Cleaning & Structuring
Removes extra whitespace, special characters

Stores cleaned data in a Pandas DataFrame

Removes duplicate questions

🔹 Categorization
Assigns categories to questions using keyword-based logic

🔹 Embedding
Converts each FAQ question into a vector embedding using all-MiniLM-L6-v2

🔹 Indexing
Uses FAISS for fast nearest-neighbor search

🔹 User Interaction
User types a question in the Streamlit chat interface

The question is embedded

The embedding queries the FAISS index

If a match is found (based on similarity score), the answer is shown

If not, the bot politely indicates it's unsure

Session state is used to track the conversation

🛠️ Setup and Installation
To run this project locally:

✅ 1. Clone or Save the Repository
Save your main file as chatbot_app.py (or any .py file).

✅ 2. Create a Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
# Activate:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


✅ 3. Install Dependencies
bash
Copy
Edit
pip install streamlit requests beautifulsoup4 pandas transformers sentence-transformers faiss-cpu torch numpy
▶️ Running the Application
Start the app by running:

bash
Copy
Edit
streamlit run chatbot_app.py
This will launch the chatbot in your default web browser.

📁 Project Structure
chatbot_app.py – Main script containing the complete logic.

load_and_process_faq_data() – Handles scraping, cleaning, deduplication, categorization

load_models_and_create_index() – Loads models & sets up FAISS index

paraphrase_answer() – (Optional) Paraphrasing using T5 model

find_best_faq_match() – Performs semantic search and threshold filtering

Streamlit UI logic – Manages chat interface, user input, and session state


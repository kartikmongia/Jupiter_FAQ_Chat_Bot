import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch

# --- 1. Data Scraping and Preprocessing ---
@st.cache_resource
def load_and_process_faq_data():
    """
    Scrapes FAQ data from Jupiter Money website, cleans it,
    removes duplicates, and categorizes questions.
    """
    url = "https://jupiter.money/contact/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"}

    try:
        response = requests.get(url, headers=headers, timeout=10) # Added timeout for robustness
        response.raise_for_status() # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from {url}: {e}")
        st.stop() # Stop the app execution if data fetching fails

    soup = BeautifulSoup(response.text, 'html.parser')
    faq_section = soup.find('ul', {'data-controller': 'faq'})

    if not faq_section:
        st.error("Could not find the FAQ section on the webpage. The website structure might have changed.")
        st.stop()

    faq_items = faq_section.find_all('li')

    faq_data = []
    for item in faq_items:
        question_tag = item.find('span')
        answer_tag = item.find('p')

        question = question_tag.get_text(strip=True) if question_tag else ''
        answer = answer_tag.get_text(strip=True) if answer_tag else ''

        if question and answer: # Only add if both question and answer are found
            faq_data.append({'question': question, 'answer': answer})

    df = pd.DataFrame(faq_data)

    def clean_text(text):
        """Removes extra whitespace and strips text."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    df['question_cleaned'] = df['question'].str.lower().str.replace(r'[^\w\s]', '', regex=True).apply(clean_text)
    df = df.drop_duplicates(subset='question_cleaned')

    def categorize(question_text):
        """Categorizes questions based on keywords."""
        if any(keyword in question_text.lower() for keyword in ['kyc', 'identity']):
            return 'KYC'
        elif any(keyword in question_text.lower() for keyword in ['reward', 'point']):
            return 'Rewards'
        elif any(keyword in question_text.lower() for keyword in ['payment', 'upi', 'transfer']):
            return 'Payments'
        elif any(keyword in question_text.lower() for keyword in ['limit', 'balance']):
            return 'Limits'
        else:
            return 'General'

    df['category'] = df['question'].apply(categorize)
    return df

# --- 2. Model Loading and FAISS Indexing ---
@st.cache_resource
def load_models_and_create_index(df_faq):
    """
    Loads the paraphrasing model and sentence embedding model,
    then creates and populates the FAISS index.
    """
    # Load Paraphrasing Model
    # Using a try-except block to handle potential issues during model loading
    try:
        tokenizer_paraphrase = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")
        model_paraphrase = AutoModelForSeq2SeqLM.from_pretrained("prithivida/parrot_paraphraser_on_T5")
    except Exception as e:
        st.error(f"Error loading paraphrasing model: {e}")
        st.stop() # Stop the app if model loading fails

    # Load Sentence Transformer Model
    try:
        model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading sentence embedding model: {e}")
        st.stop() # Stop the app if model loading fails

    # Generate Embeddings
    faq_questions = df_faq['question'].tolist()
    faq_embeddings = model_embedding.encode(faq_questions, show_progress_bar=False)

    # Create FAISS Index
    dimension = faq_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(faq_embeddings))

    return tokenizer_paraphrase, model_paraphrase, model_embedding, index

# --- 3. Paraphrasing Function (using the loaded model) ---
def paraphrase_answer(answer, tokenizer_paraphrase, model_paraphrase, num_return_sequences=1):
    """
    Generates paraphrased versions of an answer using the loaded T5 model.
    """
    text = f"paraphrase: {answer} </s>"
    encoding = tokenizer_paraphrase.encode_plus(text, padding='longest', return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    # Ensure model is in evaluation mode
    model_paraphrase.eval()

    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model_paraphrase.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            # early_stopping=True, # This flag is deprecated in newer transformers versions and can cause warnings
            num_return_sequences=num_return_sequences
        )
    return [tokenizer_paraphrase.decode(output, skip_special_tokens=True) for output in outputs]

# --- 4. Find Best FAQ Match Function ---
def find_best_faq_match(query, df_faq, model_embedding, index, threshold=0.4):
    """
    Finds the best matching FAQ answer for a given query using FAISS index.
    """
    query_embedding = model_embedding.encode([query])
    distances, indices = index.search(np.array(query_embedding), 1)

    best_distance = distances[0][0]
    best_index = indices[0][0]

    if best_distance < threshold:  # Closer distance means more similar
        answer = df_faq.iloc[best_index]['answer'] # Use original answer for bot
        # Optionally, you could paraphrase this answer for variety
        # rephrased_answer = paraphrase_answer(answer, tokenizer_paraphrase, model_paraphrase)[0]
        return answer
    else:
        return "I'm not sure about that. Could you rephrase or ask something else?"

# --- Streamlit App Layout ---
st.set_page_config(page_title="Jupiter FAQ Bot", layout="centered")

st.title("🧠 Jupiter FAQ Bot")
st.markdown("Ask me anything about Jupiter's services!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "bot", "content": "Hi! Ask me anything about Jupiter's services. Type 'exit' to stop."})

# Load data and models (cached)
with st.spinner("Loading FAQ data and AI models... This might take a moment."):
    df_faq = load_and_process_faq_data()
    tokenizer_paraphrase, model_paraphrase, model_embedding, faiss_index = load_models_and_create_index(df_faq)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Your question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get bot response
    with st.spinner("Thinking..."):
        if user_input.lower() in ['exit', 'quit']:
            bot_response = "Bye! Have a great day."
        else:
            bot_response = find_best_faq_match(user_input, df_faq, model_embedding, faiss_index)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    # Display bot response in chat message container
    with st.chat_message("bot"):
        st.markdown(bot_response)


import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()

# Load the GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key is None:
    st.error("Groq API Key is missing.")
    st.stop()  # Stop execution if key is missing

# Huggingface embedding
hf_token = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    The user will ask for recommendations on which course they should take. Based on the user's query, recommend the most relevant free courses from Analytics Vidhya's available courses (from the URL provided to you).

    Your response should include:
    - Course name
    - A brief course description
    - URL of the course

    Guidelines:
    - Only recommend free courses (exclude blogs or other types of content).
    - Ensure the recommendations are highly relevant to the user's query.
    - If multiple courses are relevant, recommend them based on similarity to the user's query, with the most relevant listed first.

    <context>
    {context}
    <context>

    User's Question: {input}
    """
)


# Function to scrape website content
def scrape_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from paragraphs and combine
        paragraphs = soup.find_all('p')
        text_content = ' '.join([para.get_text() for para in paragraphs])

        return text_content

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching website content: {e}")
        return None

# Create vector embeddings from website content
def create_vector_embedding_from_url(url):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Scrape website and get the text
        website_content = scrape_website_content(url)
        if website_content:
            # Split the content into documents (chunks)
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.create_documents([website_content])
            
            # Create vector store
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        else:
            st.error("Failed to retrieve content from the website")

st.title("🌐 RAG Website Q&A With Groq And Llama3")
st.write("**Get your perfect course in just few seconds !**")

url_input = st.text_input("Enter a website URL to use for Q&A", value ="https://courses.analyticsvidhya.com/pages/all-free-courses")
user_prompt = st.text_input("Enter your query from the website content", placeholder = "Suggest me a course for learning Python for Data Science")

if st.button("Website Embedding"):
    create_vector_embedding_from_url(url_input)
    st.write("Vector Database is ready")

import time

if user_prompt:
    # Ensure vectors are initialized before using them
    if "vectors" not in st.session_state:
        st.error("Please create vector embeddings first by clicking 'Website Embedding'.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.time()
        response = retrieval_chain.invoke({'input': user_prompt})
        print(f"Response time: {time.time() - start}")

        if 'answer' in response:
            st.write(response['answer'])

        # Handle context safely
        if 'context' in response:
            with st.expander("Website similarity Search"):
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write('------------------------')

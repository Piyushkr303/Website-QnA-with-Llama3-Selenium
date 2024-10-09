---
title: Website QnA Using Llama3 And Selenium
emoji: ⚡
colorFrom: gray
colorTo: blue
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# RAG Website Q&A With Groq and Llama3

## Overview

This application allows users to extract information from a website and receive answers to specific queries based on the website's content. It uses Retrieval-Augmented Generation (RAG) by combining document retrieval techniques with a language model for accurate, context-aware responses.

The system enables users to input a URL (such as Analytics Vidhya's free courses page) and pose a question related to the website's content. The app will respond with relevant information based on the scraped data. The primary focus is on educational content, specifically course suggestions.

## Features

- **Website scraping**: Extracts textual content from the given website URL.
- **Embeddings generation**: Converts the scraped content into vector embeddings for document retrieval.
- **Question answering**: Uses the Groq LLM (`Llama3-8b-8192`) to answer queries based on the embedded content.
- **Analytics Vidhya Course Recommendation**: Provides course details such as course name, description, curriculum, and key topics based on user queries.

## Requirements

The project relies on several libraries and requires API tokens for `Groq` and `Hugging Face`. Below are the key requirements:

- **Python 3.8+**
- **Streamlit**: For building the user interface.
- **requests**: For making HTTP requests and scraping web content.
- **BeautifulSoup (bs4)**: For parsing and extracting text from HTML.
- **Langchain**: For integrating the Groq language model and text retrieval pipeline.
- **FAISS**: For vector database creation and similarity search.
- **dotenv**: For loading API keys from environment variables.

## Setup Instructions

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Install the dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure environment variables

Create a `.env` file in the root directory with your API keys:

```bash
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

### Step 4: Run the application

You can start the Streamlit app using the following command:

```bash
streamlit run app.py
```

## Usage

1. **Enter a Website URL**: Input the URL of the website you want to scrape for the Q&A system. For example, Analytics Vidhya's free courses page.
   
2. **Click 'Website Embedding'**: This scrapes the content from the URL and generates vector embeddings for document retrieval.

3. **Ask a Question**: Input your query, such as "Suggest me a course for learning Python for Data Science". The application will retrieve relevant content and provide an answer.

4. **Explore the Context**: The system also provides a similarity search option (available as an expander), allowing you to review additional context or related content retrieved from the website.

## Main Components

1. **Scraping Content**: 
   - The `scrape_website_content()` function retrieves and processes text from the given URL.
  
2. **Embeddings Creation**: 
   - The `create_vector_embedding_from_url()` function generates vector embeddings using the Hugging Face model `all-MiniLM-L6-v2` and stores them in an FAISS vector database.

3. **LLM Integration**: 
   - The Groq language model (`Llama3-8b-8192`) is integrated to handle user queries with the help of a custom prompt template.

4. **Retrieval Chain**: 
   - The `create_retrieval_chain()` function establishes a chain combining document retrieval with Groq's LLM to generate a response based on the context retrieved from the website.

## Notes

- Ensure the website content is structured properly and includes relevant information, such as courses, to get meaningful responses.
- The system currently focuses on educational content recommendations but can be adapted for other types of information retrieval tasks.

## Future Improvements

- **Enhanced UI/UX**: Improvements can be made to the interface to offer a more engaging user experience.
- **Additional Sources**: Support for querying multiple websites or sources at once.
- **Improved Error Handling**: Provide more robust error handling for cases where the website structure or content is insufficient for scraping or retrieval.

## License

This project is licensed under the MIT License.

## Contributing

Feel free to open issues or submit pull requests for improvements.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

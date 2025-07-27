# AI Chatbot with NLP

This project is a simple, command-line-based AI chatbot built using Python and the Natural Language Toolkit (NLTK). The chatbot is capable of answering user queries based on a provided text corpus. It uses the TF-IDF algorithm to find the most relevant sentence from its knowledge base and returns it as the answer.

## Features

-   **Question & Answer:** Answers questions based on a pre-defined knowledge base (corpus).
-   **NLP-Powered:** Uses TF-IDF vectorization and Cosine Similarity to match user queries with the best possible answer.
-   **Basic Chit-Chat:** Handles simple greetings and conversational phrases like "how are you?" and "thanks."
-   **Lightweight:** Runs entirely in the terminal with minimal dependencies.
-   **Customizable:** The chatbot's knowledge can be easily expanded by modifying the text corpus.

## How It Works

The chatbot's logic follows these steps:
1.  **Corpus Loading:** The script starts with a block of text that serves as its knowledge base.
2.  **Preprocessing:** User input and corpus sentences are tokenized, converted to lowercase, and lemmatized (reducing words to their root form).
3.  **TF-IDF Vectorization:** The preprocessed text is converted into numerical vectors. TF-IDF (Term Frequency-Inverse Document Frequency) measures how important a word is to a document in a collection of documents.
4.  **Cosine Similarity:** The chatbot calculates the cosine similarity between the user's query vector and all the sentence vectors in the corpus.
5.  **Response Generation:** The sentence with the highest similarity score is selected and returned as the answer. If no relevant sentence is found (similarity score is 0), a default message is returned.

## Prerequisites

Before you begin, ensure you have Python installed on your system (version 3.6 or higher).

## Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install the required Python libraries:**
    Open your terminal or command prompt and run the following command to install NLTK and Scikit-learn.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt` file, you can install them manually)*
    ```bash
    pip install nltk scikit-learn
    ```

3.  **Download NLTK Data:**
    The first time you run the script, it will attempt to download the necessary NLTK data packages (`punkt`, `wordnet`, `punkt_tab`). If this fails, you can download them manually by running a Python interpreter:
    ```python
    >>> import nltk
    >>> nltk.download('punkt')
    >>> nltk.download('wordnet')
    >>> nltk.download('punkt_tab')
    >>> nltk.download('stopwords')
    ```

## How to Run the Chatbot

To start the chatbot, navigate to the project directory in your terminal and run the Python script:

```bash
python chatbot.py

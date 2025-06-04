# PDF Document Analysis Application

This is a Python application that uses LangChain and Gradio to analyze PDF documents. The application leverages OpenAI's language models for processing and understanding document content.

## Setup

0. Install python in macOs using command

```bash
brew install python
```

1. Clone this repository
2. Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
pip3 install -r requirements.txt
```

4. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Running the Application

To run the application:

```bash
python3 app.py
```

The application will start a local Gradio server, and you can access the interface through your web browser.

## Features

- PDF document upload and analysis
- Text extraction and processing
- Interactive Q&A with document content
- User-friendly web interface

## Dependencies

- gradio: Web interface framework
- langchain: Framework for developing applications powered by language models
- pypdf: PDF processing library
- langchain-openai: OpenAI integration for LangChain
- langchain-chroma: Vector store for document embeddings

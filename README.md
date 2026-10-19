# DocuMind AI

**A local PDF question-answering assistant powered by Retrieval-Augmented Generation (RAG).**

DocuMind AI is an AI-powered application that allows users to ask questions about PDF documents and receive answers grounded in the retrieved document passages.

The application combines semantic search, keyword-based retrieval and a locally hosted language model to provide a document-focused question-answering experience.

## Features

- **PDF ingestion:** extract text from PDF documents using PyMuPDF.
- **Sentence-aware chunking:** split document text into manageable passages while preserving sentence boundaries where possible.
- **Semantic retrieval:** use multilingual sentence embeddings to find passages related to a question.
- **Hybrid retrieval:** combine semantic search and BM25 keyword search using Reciprocal Rank Fusion (RRF).
- **Local answer generation:** generate answers with Llama 3.2 through Ollama.
- **Source traceability:** display document names and page references for retrieved passages.
- **Interactive interface:** upload PDFs, ask questions and inspect retrieved context through Streamlit.

## Architecture

1. **Document ingestion:** PDF files are uploaded and their text is extracted.
2. **Chunking:** extracted text is divided into smaller passages.
3. **Indexing:** passages are encoded into multilingual embeddings and indexed for keyword search.
4. **Retrieval:** semantic and lexical rankings are combined to identify relevant passages.
5. **Generation:** the retrieved context and user question are sent to the local language model.
6. **Presentation:** the answer and supporting passages are displayed in the Streamlit interface.

## Tech stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- NumPy
- BM25 (`rank-bm25`)
- Ollama
- Llama 3.2 (`llama3.2:1b`)

## Prerequisites

- Python installed on your computer
- Ollama installed and running
- Sufficient memory and disk space for the embedding model and language model

## Installation

Clone the repository:

```bash
git clone https://github.com/Martinien21/documind-ai.git
cd documind-ai
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the language model:

```bash
ollama pull llama3.2:1b
```

Ensure Ollama is running, then launch the application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Usage

1. Launch DocuMind AI.
2. Upload one or more PDF documents.
3. Build the document index.
4. Ask a question about the uploaded documents.
5. Review the generated answer and the retrieved source passages.

## Evaluation

The project includes basic tests for chunking and hybrid retrieval.

A next step is to build a manually annotated evaluation dataset and compare semantic retrieval, BM25 and hybrid retrieval using metrics such as Recall@k and Mean Reciprocal Rank (MRR).

Answer quality, source attribution and the model's ability to abstain when information is missing should also be evaluated.

## Limitations

- Answer quality depends on document extraction, passage retrieval and the language model.
- Scanned PDFs may require OCR if they do not contain extractable text.
- A small local language model may produce incomplete or inaccurate answers.
- Retrieved passages and citations should be checked against the original documents.
- Model downloads require an internet connection, although inference can run locally afterward.

## Privacy

Document processing and answer generation are designed to run locally. Keep confidential documents out of public repositories and verify the behavior of any external services before using sensitive data.

## Roadmap

- [ ] Evaluate retrieval quality on a curated question dataset.
- [ ] Compare semantic, lexical and hybrid retrieval.
- [ ] Improve answer completeness and handling of unanswerable questions.
- [ ] Add automated tests for retrieval and generation.
- [ ] Improve the interface and document citation display.

## License

A license has not yet been selected. Reuse and redistribution permissions will depend on the license added to this repository.

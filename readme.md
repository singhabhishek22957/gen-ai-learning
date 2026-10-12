# Generative AI Learning 🚀

Welcome to my **Generative AI learning repository**! This repository contains my Python code, experiments, and practice notebooks as I learn Generative AI, Large Language Models (LLMs), and LangChain.

I'm learning through hands-on practice and the **CampusX YouTube course**.

## 📚 Topics Covered

### 1. Python for GenAI
- Python fundamentals
- Working with libraries and packages
- Environment variables using `.env`
- Jupyter Notebook experiments

### 2. Large Language Models (LLMs)
- Introduction to LLMs
- Running local language models
- Model inference
- Prompt engineering
- Temperature and model parameters

### 3. LangChain
- Introduction to LangChain
- Models and chat models
- Prompt templates
- Runnable interfaces and LCEL
- Output parsers
- Structured output
- Chains and component composition

### 4. Document Loaders
Learning how to load different types of documents into LangChain.

- Text files
- PDF documents
- CSV files
- Excel files
- Word documents
- Web pages

### 5. Text Splitters
Learning how to divide large documents into smaller chunks for effective retrieval and processing.

- Character text splitting
- Recursive character text splitting
- Token-based splitting
- Chunk size and chunk overlap
- Metadata preservation

### 6. Retrieval-Augmented Generation (RAG)
Learning how to build applications that answer questions using external documents and retrieved context.

- Document loading
- Text splitting and chunking
- Embeddings
- Vector databases
- Similarity search
- Retrievers
- Context-based question answering
- RAG pipeline implementation

### 7. Ollama and Local LLMs
- Running open-source models locally
- Integrating Ollama with LangChain
- Using `ChatOllama`
- Model selection and configuration
- Local inference experiments

## 🛠️ Tech Stack

- **Language:** Python
- **LLM Framework:** LangChain
- **Local Model Runtime:** Ollama
- **Development:** Jupyter Notebook, VS Code
- **Version Control:** Git and GitHub

## 📂 Repository Structure

```text
gen-ai/
├── basics/                 # Python and GenAI fundamentals
├── langchain/              # LangChain practice
├── document_loaders/       # Loading documents
├── text_splitters/         # Document chunking
├── rag/                    # Retrieval-Augmented Generation
├── ollama/                 # Local LLM experiments
├── notebooks/              # Jupyter notebooks
├── .env.example            # Example environment variables
├── .gitignore              # Ignored files and folders
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

*The directory names above are an organizational suggestion. Adjust them to match the folders in your actual project.*

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

For Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create your local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Add the API keys required by your code. Never commit real API keys or credentials to GitHub.

### 5. Set up Ollama

Install Ollama from [ollama.com](https://ollama.com/).

Download a model:

```bash
ollama pull qwen2.5-coder:3b
```

Run it:

```bash
ollama run qwen2.5-coder:3b
```

## 💻 Example: LangChain with Ollama

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0.7,
)

response = llm.invoke(
    "Explain Retrieval-Augmented Generation in simple terms."
)

print(response.content)
```

Make sure Ollama is running and the selected model is downloaded before executing the example.

## 🎓 Learning Resource

**Teacher:** CampusX

[Watch the CampusX YouTube course](https://www.youtube.com/watch?v=bL92ALSZ2Cg&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0&index=12)

I'm using this course as a learning resource while practicing GenAI and LangChain concepts.

## 🎯 Learning Goals

- Build a strong foundation in Generative AI.
- Understand how LLMs work and how to use them.
- Learn LangChain components and runnable interfaces.
- Load and process documents using document loaders and text splitters.
- Understand embeddings, vector search, and RAG pipelines.
- Experiment with open-source LLMs using Ollama.
- Build practical GenAI applications through hands-on coding.

## 📈 Progress

This is an ongoing learning repository. I'll continue adding code examples, notebooks, and experiments as I progress through the topics.

## 🔐 Security

- Keep `.env` out of version control.
- Use `.env.example` without real secrets.
- Exclude `venv/`, `__pycache__/`, and downloaded model files using `.gitignore`.
- Never commit API keys, passwords, or access tokens.

## 📄 License

This repository is intended for educational and personal learning purposes. Add a license if you want to specify how others may reuse your code.

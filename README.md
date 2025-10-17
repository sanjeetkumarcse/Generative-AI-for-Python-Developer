# Generative AI Training for Python Developers

### Curriculum Breakdown

#### 1. Foundation & Setup
- Introduction to Generative AI (history, use cases, industry trends)
- Machine Learning vs. Deep Learning vs. GenAI (conceptual comparison)
- Python environment setup (Anaconda, venv, VS Code)
- Introduction to key libraries: NumPy, Pandas, Matplotlib
- **Hands-on:** Creating and activating a virtual environment, installing dependencies

**Labs:**
- Lab 1: Setup Python Virtual Environment (conda/venv)
- Lab 2: Load CSV file with Pandas, clean missing data, visualize with Matplotlib

#### 2. Core Concepts & Libraries
- Neural Networks Fundamentals (perceptron, activation functions)
- Intro to TensorFlow and PyTorch
- Transfer Learning & Pre-trained Models
- API Integrations (OpenAI GPT, Hugging Face Transformers)
- **Hands-on:** Building a Simple Chatbot

**Labs:**
- Lab 3: Implement a basic neural network in PyTorch
- Lab 4: Load a pre-trained Hugging Face model and generate text
- Lab 5: Connect to OpenAI API and build a Q&A chatbot

#### 3. Practical Applications
- Text Generation (language models, temperature, top-k/top-p sampling)
- Image Generation Basics (diffusion models overview)
- Data Preprocessing for AI models (tokenization, embeddings)
- Prompt Engineering (zero-shot, few-shot examples)
- **Hands-on:** Build a Simple Content Generator

**Labs:**
- Lab 7: Generate AI images using Stable Diffusion API
- Lab 8: Experiment with prompt engineering for better outputs

#### Case Study (Team Activity)
**Title:** AI-Powered Marketing Content Assistant  
**Objective:**
- Preprocess real-world dataset of marketing prompts
- Use pre-trained model to generate ad copies & blog headlines
- Evaluate outputs for creativity and coherence
- Present improvements using prompt tuning

---

#### 4. Advanced Frameworks & Tools
**Topics:**
- Deep Dive: LangChain & LlamaIndex
- Vector Databases: Pinecone, Weaviate, Chroma
- API Rate Limiting & Error Handling
- **Hands-on:** Document Q&A System

**Lab Setup & Activities:**
- Install langchain, llama-index, chromadb, openai
- Configure Chroma or Pinecone database
- Ingest and embed sample documents
- Build RetrievalQA chain and test with multiple queries

#### 5. Custom Model Development
**Topics:**
- Fine-Tuning Pre-trained Models
- Transfer Learning Techniques
- Model Evaluation: Perplexity, BLEU, ROUGE
- **Hands-on:** Domain-Specific Fine-tuning

**Lab Setup & Activities:**
- Install transformers, datasets, accelerate, evaluate
- Load custom dataset and tokenize
- Fine-tune GPT-2 or FLAN-T5 on sample corpus
- Evaluate results and compare metrics

#### 6. RAG Systems & Knowledge Bases
**Topics:**
- Retrieval-Augmented Generation Architecture
- Chunking Strategies for Large Documents
- Embedding Models & Similarity Search
- **Hands-on:** Build Enterprise Knowledge Assistant

**Lab Setup & Activities:**
- Install sentence-transformers, pinecone-client
- Chunk sample PDFs using LangChain tools
- Store embeddings in Chroma or Pinecone
- Integrate retrieval pipeline with LLM for Q&A

#### 7. Advanced Applications
**Topics:**
- Multi-modal AI: Text, Image, Audio Pipelines
- Agent-based Systems: AutoGen, CrewAI, LangGraph
- Model Context Protocol (MCP) & MCP Servers
- Agent2Agent Communication (AutoGen, Camel, ReAct)
- Workflow Automation
- **Hands-on:** Build AI-powered Automation Tool

**Lab Setup & Activities:**
- Install autogen, crewai, langgraph, tts
- Create multi-agent collaboration workflow
- Compare CrewAI vs LangGraph with same task
- Automate simple workflow (e.g., summarize + send email)

#### 8. Production Deployment
**Topics:**
- Containerization with Docker
- REST API Development with FastAPI
- Monitoring & Logging
- **Hands-on:** Deploying AI Application

**Lab Setup & Activities:**
- Install fastapi, uvicorn, prometheus-client
- Write Dockerfile and containerize app
- Expose REST endpoints
- Implement request logging and monitoring dashboard

#### Case Study: Enterprise Knowledge Assistant
**Objective:** Build an assistant that answers HR policy questions from internal documents  
**Dataset:** Sample HR policy PDFs (leave policy, reimbursement, WFH rules)  
**Tasks:**
- Preprocess & chunk PDFs
- Generate embeddings and build retriever
- Build QA interface with Streamlit
- Evaluate answers for accuracy & coverage

#### Capstone Projects
**Capstone Project 1:** GenAI Content Creation Suite
- Chatbot interface  
- Text generator for blog intros and product descriptions  
- Simple image generator using Stable Diffusion API  
- UI with Streamlit or Flask for end-to-end workflow  
- API & Deployment – Expose via FastAPI, containerize with Docker  
- Monitoring – Collect logs and performance metrics  

**Capstone Project 2:** AI-powered Workflow Automation Suite
- RAG-based Knowledge Assistant – Q&A from documents  
- Agentic Workflow – Automate form-filling, email writing  
- Multi-modal Support – Generate reports with images  
- API & Deployment – Expose via FastAPI, containerize with Docker  
- Monitoring – Collect logs and performance metrics  

#### Evaluation Parameters
| Criteria | Weight |
|----------|--------|
| Functional Completeness | 25% |
| Model Performance (RAG + LLM) | 20% |
| Code Quality & Best Practices | 15% |
| Deployment & Scalability | 15% |
| Innovation/Extra Features | 10% |
| Documentation & Reporting | 10% |
| Presentation & Demo Clarity | 5% |

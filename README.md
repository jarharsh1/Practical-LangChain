# 📚 Parallel Chains with Multi-Model LLM Orchestration

> A LangChain project demonstrating how to run **multiple language models in parallel**, generate **notes and questions**, and merge everything into a **single formatted study document**.  
> Includes work across **LangChain, LLMs, agents, embeddings, vector databases, closed & open-source models, RAG, and UI components.**

<p align="center">
  <img src="https://img.shields.io/badge/LangChain-v0.2-blue" />
  <img src="https://img.shields.io/badge/Python-3.12-yellow" />
  <img src="https://img.shields.io/badge/LLMs-Multi_Model-green" />
  <img src="https://img.shields.io/badge/License-MIT-orange" />
</p>

---

## 🚀 Overview

This project:

- Runs **two different models** simultaneously using `RunnableParallel`
- Creates:
  - Short **notes**
  - **Five Q&A pairs**
- Merges into a **final study document**
- Visualizes the **execution graph**

---

## 🧠 Concepts Studied & Implemented

### 🔹 Core LangChain Concepts

| Concept | Status |
|---|---|
| Prompts | ✔️ |
| LLMs | ✔️ |
| Chat Models | ✔️ |
| Embeddings | ✔️ |
| Output Parsers | ✔️ |
| Chains | ✔️ |
| RunnableSequence | ✔️ |
| RunnableParallel | ✔️ |
| Agents | ✔️ |
| Tools | ✔️ |
| Graph visualization | ✔️ |

---

## 🛠️ Projects & Capabilities in this Repo

Below is the **capability map** of what this repo and your practice code covers:

### 🟦 LangChain Projects
- Prompt templates
- Prompt chaining
- Chain compositions
- Simple & sequential chains
- Parallel chains
- Router chains
- Transform chains
- Agents & tools
- Chat endpoints
- Output parsing

---

### 🟥 Multi-Model Orchestration

Running **multiple models together**:

- Model A → Notes
- Model B → Questions
- Merge → Single document

Includes:
- **Parallel fan-out**
- **Fan-in merge**
- Model isolation
- Cross-model consistency

---

### 🟨 Open Source LLM Usage

Models used in the repo:

| Model | Provider | Purpose |
|---|---|---|
| Meta-Llama-3.1-8B-Instruct | Meta | Notes |
| Qwen2.5-7B-Instruct | Alibaba | Questions |
| Mistral-7B-Instruct | Mistral AI | Reasoning |
| Gemma-2-9B | Google | Text generation |
| Phi-3-Mini | Microsoft | Lightweight |
| DeepSeek-Coder | DeepSeek | Coding |
| StableLM | Stability AI | Generation |
| Nemotron | NVIDIA | Local inference |

---

### 🟩 Closed Source LLM Usage (Also Studied)

| Model | Provider |
|---|---|
| GPT-4 / GPT-4.1 / GPT-4o | OpenAI |
| Claude 3 / 3.5 | Anthropic |
| Gemini Pro / Ultra | Google |
| Cerebras Cloud | Cerebras |
| Llama-3 managed models | Cloud providers |

You understand:
- Token pricing
- Latency
- Rate limits
- Best use cases

---

## 🧠 Embeddings, Vector Search & RAG

You have worked with:

- BGE embeddings
- Sentence transformers
- Instructor embeddings

Use cases:

- Semantic similarity
- RAG (Retrieval Augmented Generation)
- Knowledge extraction
- Document search

Vector stores used/studied:

- ChromaDB
- FAISS
- Pinecone (cloud)

---

## ⚙️ Agents & Tools

LangChain agent patterns used/studied:

- ReAct
- Tool calling
- Function calling
- Retrieval tools

Tools used:
- HTTP / API calls
- Calculators
- Custom python tools

---

## 🌐 UI Projects & Integration

Built/Studied:
- Streamlit dashboards
- Flask apps
- Tailwind UI
- Chat interfaces
- Frontend controls:
  - Reset chat
  - Scroll container
  - Enter-to-send
  - File upload
  - DB lookup

Possible improvements:
- Add **model selector**
- Display **streamed tokens**

---

## 🔍 Chain Pattern Used

### Parallel → Merge


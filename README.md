## Why I Built This
I'm Shashank Shandilya, a software engineer working on production-grade agentic database support systems.

I built NexusOps to demonstrate end-to-end AI engineering skills: from training
custom PyTorch models to orchestrating multi-agent workflows with LangGraph
and deploying on Kubernetes with full CI/CD.

The problem space (autonomous SRE) genuinely interests me because in my current organisation I see multiple teams working on migrating databases from on premises to cloud environments. This process is biring up new challenges and concerns. I believe that an agentic system like this can help solve these problems more efficiently and effectively. Additionally I am very interested in the intersect of AI and DevOps and wanted to explore this space further.

# 🚀 NexusOps — Autonomous AI Site Reliability Engineer

> An autonomous multi-agent system that monitors cloud infrastructure, detects anomalies, diagnoses root causes, proposes fixes, and executes remediation actions with human approval.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-EKS-326CE5?logo=kubernetes)](https://kubernetes.io)

---

## 🧠 Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │          NEXUSOPS SUPERVISOR AGENT           │
                    │          (LangGraph StateGraph)              │
                    │   Receives alerts → routes → synthesizes     │
                    └──────────────┬──────────────────────────────┘
                                   │
          ┌────────────┬───────────┼───────────┬────────────┐
          ▼            ▼           ▼           ▼            ▼
   ┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
   │  SENTINEL   │ │DETECTIVE │ │ ORACLE   │ │ SURGEON  │ │  SCRIBE      │
   │  AGENT      │ │  AGENT   │ │  AGENT   │ │  AGENT   │ │  AGENT       │
   │             │ │          │ │          │ │          │ │              │
   │Monitors     │ │Root cause│ │Predicts  │ │Executes  │ │Generates     │
   │metrics with │ │analysis  │ │future    │ │approved  │ │post-incident │
   │PyTorch LSTM │ │via RAG   │ │failures  │ │remediatio│ │reports &     │
   │anomaly      │ │(LangChain│ │LSTM model│ │n actions │ │runbooks      │
   │detection    │ │+ pgvector│ │          │ │(dry-run) │ │(pgvector)    │
   └─────────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘
```

## 📦 Tech Stack

| Layer | Technologies |
|---|---|
| **AI / ML** | PyTorch, TensorFlow/Keras, Hugging Face, LangChain, LangGraph |
| **Cloud** | AWS (EKS, S3, ECR, RDS, CloudWatch), GCP (Vertex AI, BigQuery) |
| **Infrastructure** | Kubernetes, Docker, Terraform, Helm |
| **MLOps** | MLflow, LangSmith, Prometheus, Grafana |
| **CI/CD** | GitHub Actions |
| **Backend** | FastAPI, Redis, PostgreSQL + pgvector, Qdrant |

## 🗂️ Project Structure

```
nexusops/
├── agents/                    # All 5 specialized agents
│   ├── sentinel/              # PyTorch anomaly detection agent
│   ├── detective/             # RAG-based root cause analysis agent
│   ├── oracle/                # Predictive failure agent
│   ├── surgeon/               # Remediation execution agent
│   └── scribe/                # Report generation agent
├── supervisor/                # LangGraph orchestration
├── ml/                        # PyTorch & TensorFlow models
│   ├── pytorch/               # LSTM autoencoder
│   └── tensorflow/            # Log pattern classifier
├── simulator/                 # Synthetic metrics & log generator
├── api/                       # FastAPI gateway
├── infra/                     # Terraform & Helm charts
│   ├── terraform/
│   └── helm/
├── .github/workflows/         # CI/CD pipelines
├── docker/                    # Dockerfiles per service
├── tests/                     # Unit + integration tests
└── notebooks/                 # Jupyter exploration notebooks
```

## 🚀 Quick Start (Phase 1 — Local)

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd nexusops

# 2. Create virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure env
cp .env.example .env
# Edit .env with your API keys

# 5. Start local services (PostgreSQL, Redis)
docker-compose up -d

# 6. Run the metrics simulator
python -m simulator.generator

# 7. Train the PyTorch anomaly model
python -m ml.pytorch.train

# 8. Train the TensorFlow log classifier
python -m ml.tensorflow.train

# 9. Start the FastAPI gateway
uvicorn api.main:app --reload --port 8000

# 10. (Optional) Launch the full agent graph
python -m supervisor.run
```

## 📊 Phases

- [x] **Phase 1** — Foundation: PyTorch + TensorFlow models, FastAPI, local simulator
- [ ] **Phase 2** — Multi-Agent: Full LangGraph graph with all 5 agents
- [ ] **Phase 3** — Cloud: EKS, RDS, Vertex AI, Terraform
- [ ] **Phase 4** — CI/CD: GitHub Actions, MLflow, model A/B testing
- [ ] **Phase 5** — Polish: Dashboard, demo video, blog post

---

*Built as a capstone project to demonstrate production-grade AI engineering skills.*

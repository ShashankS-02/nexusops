## Why I Built This

I'm Shashank Shandilya, a software engineer working on production-grade agentic database support systems.

I built NexusOps to demonstrate end-to-end AI engineering skills: from training custom PyTorch models to orchestrating multi-agent workflows with LangGraph and closing the loop with RAG based incident memory.

The problem space (autonomous SRE) genuinely interests me because in my current organisation I see multiple teams working on migrating databases from on premises to cloud environments. This process is bringing up new challenges and concerns. I often see multiple teams escalating issues related to this to the DBA team and the production readiness team and there is always a need for more support. I believe that an agentic system like this can help solve these problems more efficiently and effectively. Additionally I am very interested in the intersect of AI and DevOps and wanted to explore this space further.

# 🚀 NexusOps - Autonomous AI Site Reliability Engineer

> An autonomous multi-agent system that monitors cloud infrastructure, detects anomalies, diagnoses root causes, proposes fixes, and executes remediation actions with human approval.

**🌐 Live demo:** **[nexusops-olive.vercel.app](https://nexusops-olive.vercel.app)** — interactive landing site + the operator console (Dashboard / Incidents / Pipeline / Agents). The console runs on representative mock data in production when no backend is connected.

## Demo

![NexusOps full pipeline demo](/docs/gifs/demo.gif)

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-K3s-blue?logo=kubernetes)](https://k3s.io)
[![Helm](https://img.shields.io/badge/Helm-Chart-purple?logo=helm)](https://helm.sh)
[![CI](https://github.com/ShashankS-02/nexusops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/ShashankS-02/nexusops/actions)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=nextdotjs)](https://nextjs.org)
[![Vercel](https://img.shields.io/badge/Vercel-Live-black?logo=vercel)](https://nexusops-olive.vercel.app)

---

## 🧠 Architecture

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                         NEXUSOPS SUPERVISOR                            │
   │                        (LangGraph StateGraph)                          │
   └────────────────────────────────────────────────────────────────────────┘
                                   │
                  --------------------------------------------------
                 |              SENTINEL                            |
                 |      Trained on 5,000 synthetic sequences        |
                 | AUROC: 0.9933, Precision: 1.00 at threshold 0.60 |
                 |                                                  |
                 ---------------------------------------------------
                       ┌───────────┘
                       ▼
              ┌─────────────┐   ┌──────────┐   ┌───────────┐
              │  DETECTIVE  |-->|  ORACLE  |-->|  SURGEON  |
              │ (Root Cause)│   │ (Predict)│   │(Remediate)│
              └─────────────┘   └──────────┘   └───────────┘
                                                     │
                                     ┌---------------┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │  HUMAN APPROVAL  │
                            │(interrupt_before)│
                            └──────────────────┘
                                    |
                        ┌───────────┼───────────┐
                        ▼           ▼           ▼
               ┌──────────┐   ┌──────────┐   ┌──────────────┐
               │ SURGEON' │   │  SURGEON'│   │   SCRIBE     │
               │ (execute)│   │ (failure)│   │ (Report)     │
               └──────────┘   └──────────┘   └──────────────┘
                                      │
                                      ▼
                                    [END]
```

## 📦 Tech Stack

| Layer       | Technologies                                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------------- |
| **AI / ML** | PyTorch 2.x, TensorFlow/Keras 2.16, LangChain, LangGraph                                                 |
| **LLMs**    | OpenAI GPT-4o-mini, Groq Llama-3.3-70b                                                                   |
| **MLOps**   | MLflow, LangSmith, Prometheus, Grafana                                                                   |
| **Backend** | FastAPI, Redis, PostgreSQL + pgvector, Qdrant, HuggingFace Sentence Transformers, SQLite (checkpointing) |
| **DevOps**  | Docker, Docker Compose, Kubernetes (K3s), Helm                                                           |
| **Frontend**| Next.js 16, React 19, TypeScript, Tailwind CSS v4, shadcn/ui, Framer Motion, Recharts — deployed on Vercel |

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
│   ├── tensorflow/            # Log pattern classifier
│   └── baselines/             # Z-score & static threshold baselines
├── charts/nexusops/           # Helm chart for K8s deployment
├── k8s/                       # Sample workloads, stress tests & Grafana dashboards
├── scripts/                   # Demo & Prometheus export scripts
├── db/                        # Qdrant vector store operations
├── nexusops/                  # Core config & shared utilities
├── simulator/                 # Synthetic metrics & log generator
├── api/                       # FastAPI gateway
├── frontend/                  # Next.js web UI (Phase 5)
│   └── src/
│       ├── app/(marketing)/   # Public landing site
│       ├── app/(app)/         # Operator console: dashboard, incidents, pipeline, agents
│       ├── app/api/           # Route handlers proxying to the FastAPI backend
│       ├── components/        # UI kit, dashboard, layout & marketing components
│       └── lib/, hooks/       # API client, schema adapters, polling hook
├── .github/workflows/         # CI/CD pipelines
└── tests/                     # Unit + integration tests
```

## 🔍 LangSmith Traces

<img src="/docs/screenshots/agent_graph.png" width="600" alt="Agent execution graph">

_Two LangGraph runs for one incident. The pipeline pauses at the HITL gate after Oracle, then resumes after human approval. This is LangGraph's `interrupt_before` checkpoint pattern in action._

<img src="/docs/screenshots/detective_input.png" width="600" alt="Detective agent input">

_Detective agent receives the full alert context (pod name, anomaly score, triggered metrics) and any similar past incidents retrieved from Qdrant via RAG._

<img src="/docs/screenshots/detective_output.png" width="600" alt="Detective agent output">

_Detective returns structured JSON that contains root cause hypothesis with confidence score. This is fed directly into Oracle for blast radius prediction._

## 📊 MLflow Metrics

<img src="/docs/screenshots/MLFlow_metrics.png" width="700" alt="MLflow training metrics">

_PyTorch LSTM Autoencoder - train/val loss convergence, learning rate schedule (ReduceLROnPlateau), and AUROC 0.99 on held-out anomaly detection._

## 🖥️ Grafana Dashboard

<img src="/docs/screenshots/grafana_dashboard.png" width="700" alt="Grafana K8s monitoring dashboard">

_Live Kubernetes cluster monitoring - per-pod CPU and memory usage, network I/O, pod status, container restarts, and node-level resource gauges. Dashboard JSON is version-controlled at [`k8s/grafana/nexusops-dashboard.json`](k8s/grafana/nexusops-dashboard.json)._

## 📈 Real-Data Benchmark (K3s Cluster)

Trained the LSTM on **4,567 real Prometheus metric points** exported from a local K3s cluster running `payment-service`, `order-service`, and a stress-test workload with varied load phases (light → medium → heavy → cool down).

| Detector         | AUROC      | Precision  | Recall | F1     |
| ---------------- | ---------- | ---------- | ------ | ------ |
| Z-Score Baseline | **0.8354** | 0.9583     | 0.7419 | 0.8364 |
| LSTM Autoencoder | 0.7542     | **1.0000** | 0.3226 | 0.4878 |
| Static Threshold | 0.6296     | 1.0000     | 0.3226 | 0.4878 |

**Key takeaways:**

- On limited real data (~4.5K points), the z-score baseline outperforms the LSTM in overall AUROC
- The LSTM achieves **perfect precision** (zero false positives) — critical for production SRE where alert fatigue is a real problem
- With more temporal diversity (diurnal cycles, multi-day patterns), the LSTM's sequential pattern learning is expected to close the gap

_On synthetic data (5,000 sequences with injected anomalies), the LSTM achieves AUROC 0.9933 — see [ml/baselines/](ml/baselines/) for the full benchmark code._

## 🚀 Quick Start (Phase 1 — Local)

```bash
# 1. Clone and enter
git clone https://github.com/ShashankS-02/nexusops.git
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

# OR run the setup script to get it working on your system
./setup_phase1.sh
```

## 🚀 Quick Start (Phase 4 — Kubernetes)

```bash
# 1. Start a local K3s cluster (macOS)
brew install colima helm kubectl
colima start --kubernetes --cpu 4 --memory 8 --disk 30

# 2. Deploy the monitoring stack (Prometheus + Grafana)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# 3. Deploy NexusOps via Helm
helm install nexusops ./charts/nexusops \
  --set secrets.groqApiKey=$GROQ_API_KEY \
  --set secrets.langchainApiKey=$LANGCHAIN_API_KEY

# 4. Access Grafana and Prometheus
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80 &
kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090 &
# Grafana: http://localhost:3000 (admin / prom-operator)
# Prometheus: http://localhost:9090

# 5. Export real Prometheus metrics and retrain
python -m scripts.export_prometheus_metrics --hours 2
python -m ml.pytorch.train --data-source data/prometheus_export/metrics_combined.csv

# 6. Benchmark LSTM vs baselines
python -m ml.baselines.zscore_detector --data-source data/prometheus_export/metrics_combined.csv
```

## Example Output

A real incident run `payment-service` in `prod` with CPU at 97%, memory at 91%, latency at 5860ms, error rate at 20.5%.

**Sentinel detects the anomaly:**

> PyTorch LSTM anomaly score: **0.784** (threshold 0.60) -> incident `15460a2f` created, LangGraph pipeline triggered

**Detective (root cause analysis):**

> "The payment-service pod is likely experiencing a high error rate due to a software bug or misconfiguration, causing it to fail requests and trigger the HighErrorRate metric. This is also driving increased CPU and memory usage, as well as request latency."

**Blast radius:**

> "If left unaddressed, this could impact order processing, user accounts, and customer support — potentially leading to failed transactions and revenue loss."

**Surgeon proposes 3 remediation actions — pipeline pauses for human approval:**

> 1. `[MEDIUM RISK]` `kubectl rollout undo deployment payment-service -n prod` - rollback to last known good version
> 2. `[LOW RISK]` `kubectl scale deployment payment-service --replicas=2 -n prod` - scale out to absorb load
> 3. `[LOW RISK]` `kubectl delete pod -l app=payment-service -n prod` - force pod restart to clear transient state

**Human approves actions 1 & 2 via API:**

```bash
curl -X POST http://localhost:8000/api/v1/incidents/15460a2f/approve \
  -d '{"approved": true, "approver": "on-call-engineer", "approved_action_indices": [0, 1]}'
```

**Scribe generates post mortem (stored in Qdrant for future RAG lookups):**

```
# Incident Report: 15460a2f
## Summary
payment-service / prod - anomaly score 0.784 - triggered metrics:
cpu_usage_percent, memory_usage_percent, request_latency_ms, error_rate_percent

## Root Cause
Software bug or misconfiguration caused a high error rate, which in turn drove
elevated CPU/memory usage and request latency. Confidence: 80%.

## Impact
Order processing, user accounts, and customer support - potential revenue impact
if not resolved within ~30 minutes of trigger time.

## Remediation
1. kubectl rollout undo deployment payment-service -n prod  [EXECUTED]
2. kubectl scale deployment payment-service --replicas=2 -n prod  [EXECUTED]

## Prevention
- Add automated rollback policy for HighErrorRate > 15% sustained over 5 minutes
- Set HPA min-replicas to 2 for payment-service to prevent single-pod bottleneck
- Integrate canary deploys to catch regressions before full rollout
```

## 🖥️ Web UI (Phase 5)

A **Next.js 16** web app (deployed on Vercel) wraps the whole system. It has two halves:

**Marketing landing site (`/`)** — a self-contained explainer of what NexusOps is and how it works: an interactive, cursor-reactive "sensor-field" hero with a live mission-control console, the incident lifecycle timeline, the five-agent showcase, an architecture diagram, the self-improving RAG loop, the human-in-the-loop safety model, and the tech stack. Fully static — works with no backend.

**Operator console (`/dashboard`, `/incidents`, `/pipeline`, `/agents`)** — the live control surface:

- **Dashboard** — animated metric cards, a status-driven agent pipeline, the anomaly-score timeline, and a recent-incidents feed.
- **Incidents** — searchable, status-filterable list with expandable detail (root cause, blast radius, proposed `kubectl` actions) and a one-click **HITL approve**.
- **Pipeline** — per-incident view that derives each LangGraph stage's state from the incident's real status.
- **Agents** — live per-agent status, derived stats, and recent activity.
- **Global command search** (`/` hotkey), a real notification bell, and a system/account menu — all polling live data.

The console talks to FastAPI through Next.js route handlers (`/api/*`) that normalize the backend schema. When no backend is connected (e.g. the public Vercel demo) it runs in **Demo Mode** — a self-contained, in-browser simulation that spawns incidents over time and advances each one through the full lifecycle (`detected → analyzing → awaiting_approval → executing → resolved`) on a clock, so visitors can watch the pipeline illuminate stage-by-stage. There's a **"Trigger incident"** button to spawn one on demand, and the HITL **Approve** action pushes an incident through remediation. A **Demo** badge labels it as simulated. Plug in a real backend and the same UI switches to live data unchanged.

**Design:** a deliberately non-generic "mission-control" system — ink canvas, hairline borders, a single electric-lime signal accent, Space Grotesk / JetBrains Mono type, and a film-grain + grid texture.

```bash
# Run the web UI locally (from repo root)
cd frontend
npm install
npm run dev            # http://localhost:3000

# Point the console at a running backend (optional)
# BACKEND_URL defaults to http://localhost:8000
```

Deploy on Vercel with **Root Directory = `frontend`** and Framework Preset **Next.js** — the landing site + Demo-Mode console deploy with **zero config** (Demo Mode is on by default when no backend is configured). For a fully live console, set `BACKEND_URL` (+ `NEXT_PUBLIC_BACKEND_URL`) to a hosted FastAPI URL, which automatically disables Demo Mode. To force either mode, set `NEXT_PUBLIC_DEMO_MODE=1` or `=0`.

## 📊 Phases

- [x] **Phase 1** — Foundation: PyTorch + TensorFlow models, FastAPI gateway, synthetic simulator
- [x] **Phase 2** — Multi-Agent: LangGraph orchestration, HITL approval, RAG memory loop
- [x] **Phase 3** — Containerization: Dockerfile, multi-stage builds, docker-compose for full stack
- [x] **Phase 4** — Kubernetes: Helm chart, K3s deployment, Prometheus + Grafana, real-data benchmark
- [x] **Phase 5** — Web UI: Next.js marketing site + operator console (dashboard, incidents, pipeline, agents), deployed on Vercel

## Limitations

- On real Prometheus data (~4.5K points), the LSTM autoencoder (AUROC 0.75) is outperformed by a z-score baseline (AUROC 0.84). The LSTM does achieve perfect precision (zero false positives), but needs more temporal diversity to leverage its sequential pattern learning.
- The real-data evaluation uses synthesized anomaly injections on top of normal Prometheus exports, since organic production incidents are rare and hard to label.
- Latency and error rate metrics are derived proxies (correlated with CPU/memory) since the sample nginx workloads don't expose native request metrics.

## Future Work

- Collect multi-day Prometheus exports with diurnal traffic cycles to give the LSTM enough temporal context to outperform statistical baselines.
- Add a ServiceMonitor CRD and instrument the sample workloads with Prometheus client libraries to capture real request latency and error rate metrics.
- Extend the Helm chart with HPA (Horizontal Pod Autoscaler) and PodDisruptionBudgets for production-grade resilience.
- Connect the deployed Vercel console to a hosted FastAPI backend (it currently falls back to mock data in production) for a fully live, end-to-end demo.

---

_Built as a capstone project to demonstrate production-grade AI engineering skills._

# Multi-Agent Graph-RAG Fraud Intelligence Platform

AI-powered insurance fraud investigation platform combining predictive machine learning, Neo4j graph analytics, Graph-RAG, LangGraph multi-agent orchestration, Gemini-based report generation, LLM evaluation, runtime observability, and FastAPI.

## Architecture

    Insurance Claims
          |
          +--------------------+
          |                    |
          v                    v
    XGBoost Fraud Model    Neo4j Graph
          |                    |
          |              Graph Features
          |              Community Detection
          |                    |
          +---------+----------+
                    |
                    v
                Graph-RAG
                    |
                    v
            LangGraph Agents
            |       |       |
          Fraud   Graph    RAG
            \       |       /
             +------v------+
                    |
                    v
           Gemini Report Agent
                    |
                    v
        Grounded Investigation
                    |
                    v
                 FastAPI

## Current Implementation

### Fraud ML

- 30,000 insurance claims
- Leakage-safe train/validation/test split
- Class-weighted XGBoost baseline
- Graph-enhanced XGBoost model
- SHAP dependency for model explainability
- Untouched test-set evaluation
- Threshold analysis

### Model Results

Baseline XGBoost:

- PR-AUC: 0.293
- ROC-AUC: 0.747

Location-aware graph-enhanced XGBoost:

- PR-AUC: 0.301
- ROC-AUC: 0.751

These metrics are from the current project's untouched test-set evaluations.

## Knowledge Graph

Neo4j currently contains:

- 30,000 Policy nodes
- 30,000 Claim nodes
- 27,214 IncidentLocation nodes
- Policy-to-claim relationships
- Claim-to-location relationships
- Shared-location claim relationships
- Incident type nodes
- Collision type nodes
- Severity nodes
- Incident-state nodes
- Louvain community detection

Graph-derived features use training-only context to reduce leakage risk.

## RAG

Current RAG stack:

- Sentence Transformers
- ChromaDB
- LangChain
- Semantic retrieval
- Graph-RAG evidence fusion

The current demonstration corpus contains synthetic internal investigation guidance created specifically for this project.

## Multi-Agent Investigation

LangGraph orchestrates:

- Fraud scoring agent
- Graph investigation agent
- RAG evidence agent
- Intent router
- Full investigation workflow
- Grounded evidence synthesis

Supported routes:

    fraud
    graph
    rag
    investigation

The full investigation route executes the specialist evidence pipeline:

    Fraud Agent
         |
         v
    Graph Agent
         |
         v
    RAG Agent
         |
         v
    Grounded Report

## LLM

Current provider:

- Google Gemini API
- google-genai
- Configurable model through .env
- Current local configuration uses gemini-3.5-flash-lite

The report agent is designed to:

- use only supplied evidence
- distinguish model signals from factual evidence
- treat retrieved guidance as guidance rather than proof
- report missing or unavailable evidence
- avoid definitive fraud determinations

## LLMOps

Current evaluation coverage includes:

- Router accuracy: 6/6 (100%)
- Evidence completeness: 6/6 (100%)
- LLM report-quality checks
- Claim ID presence
- Required report sections
- Model risk-signal reporting
- Safety checks against definitive fraud claims
- Evidence limitations checks
- Model identification
- Request latency
- Input/output/total token usage

Observed runtime example:

    Model: gemini-3.5-flash-lite
    Latency: approximately 3994 ms
    Input tokens: 1431
    Output tokens: 635
    Total tokens: 2066

These are observed values from a test request and are not production performance guarantees.

## FastAPI

Endpoints:

    GET  /health
    POST /investigate
    GET  /docs
    GET  /openapi.json

Example request:

    curl -X POST http://localhost:8000/investigate \
      -H "Content-Type: application/json" \
      -d '{
        "claim_id": "CLM_POL100000",
        "question": "Investigate this claim for potential fraud risk."
      }'

Example response structure:

    claim_id
    route
    evidence
    report

## Local Setup

### Create Python Environment

    python3 -m venv .venv
    source .venv/bin/activate

### Install Dependencies

    pip install -r requirements-core.txt
    pip install -r requirements-graph.txt
    pip install -r requirements-rag.txt
    pip install -r requirements-llm.txt
    pip install -r requirements-api.txt
    pip install -r requirements-test.txt

### Environment Variables

Create a local .env file containing:

    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_password

    GEMINI_API_KEY=your_api_key
    GEMINI_MODEL=gemini-3.5-flash-lite

Never commit .env.

A safe template is provided in:

    .env.example

## Neo4j

The local Neo4j service is defined in:

    docker/neo4j-compose.yml

## Run the API

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Open the interactive API documentation at:

    http://localhost:8000/docs

## Health Check

    curl http://localhost:8000/health

Expected response:

    {"status":"ok","service":"fraud-intelligence-api"}

## Tests

Run the API smoke and contract tests:

    pytest -q tests/test_api.py tests/test_investigate_api.py

Current result:

    3 passed

## Workflow Evaluation

Run:

    python -m src.evaluation.evaluate_workflow

Current result:

    Route accuracy:    6/6 (100.00%)
    Evidence accuracy: 6/6 (100.00%)

    WORKFLOW EVALUATION: PASS

## LLM Evaluation

Run:

    python -m src.evaluation.evaluate_llm

The evaluator checks:

- report generation
- claim ID presence
- required sections
- fraud risk signal discussion
- absence of definitive fraud determination
- evidence limitations

## Example Investigation

Current demonstration claim:

    CLM_POL100000

The integrated pipeline retrieves:

- fraud model risk signal
- policy and claim facts
- Neo4j graph evidence
- RAG evidence
- Gemini-generated investigator-oriented report
- LLM runtime metrics

Example model signal:

    Fraud risk score: 0.7372
    Threshold:        0.5
    Flagged:          true

The system treats the score as an investigation signal rather than proof of fraud.

## Repository Structure

    configs/
    data/
      evaluation/
      external/
      processed/
      rag/
      raw/
    docs/
    docker/
    notebooks/
    src/
      agents/
      api/
      config/
      data/
      evaluation/
      features/
      graph/
      models/
      rag/
      router/
      utils/
    tests/
    requirements-*.txt

## Engineering Principles

The project emphasizes:

- leakage-safe evaluation
- reproducible pipelines
- separation of evidence sources
- grounded LLM generation
- explicit model-risk limitations
- testable agent routing
- API-level validation
- runtime observability

## Current Status

The project currently has a working local:

    ML
    +
    Neo4j Graph
    +
    Graph-RAG
    +
    LangGraph Multi-Agent Workflow
    +
    Gemini LLM
    +
    LLM Evaluation
    +
    Runtime Observability
    +
    FastAPI

Additional production-oriented components such as broader evaluation benchmarks, drift monitoring, bias analysis, multimodal evidence, cloud deployment, and full CI/CD are future extensions and are not claimed as completed functionality here.

# FoodLoop AI Service 🥑⚡

The **FoodLoop AI Service** is a high-performance, production-ready Python microservice designed to handle intelligent inventory monitoring, risk assessment, multilingual historical pricing ingestion & vector retrieval, and dynamic pricing recommendation orchestration for the FoodLoop platform.

---

## 🏛️ System Responsibilities & Boundaries

> [!IMPORTANT]
> **Strict Architectural Boundary Statement**:
> The Python AI Service is a **recommendation-only and decision-support service**.
> 
> **The .NET Backend Microservice strictly owns:**
> - Final price calculation & monetary rounding
> - Price-floor enforcement
> - Database price updates & persistent transaction storage
> - Historical event persistence & authoritative source of truth
> - Owner approval workflow & UI persistence
> - Autonomous execution authorization & transaction dispatch
> - User authentication & authorization
> 
> **The Python AI Service strictly owns:**
> - Inventory Monitoring Agent & risk classification
> - Pricing Agent recommendation orchestration
> - Deterministic operational signals (Inventory coverage, demand ratio, expiry pressure)
> - Vector search & historical pricing knowledge ingestion & retrieval
> - Multilingual Arabic + English semantic retrieval using `BAAI/bge-m3` local embeddings (1024-d)
> - Deterministic AI-side operating-mode policy interpretation (`APPROVAL_REQUIRED` vs `AUTOMATIC_EXECUTION_ELIGIBLE`)
> - Concise, auditable recommendation rationale (0–15% AI recommendation boundary)

---

## 🏗️ Architecture Overview

The microservice is built using **FastAPI**, **Pydantic v2**, **LangGraph**, **LangChain**, **PyTorch**, **SentenceTransformers**, and **Qdrant Vector Database**.

```text
                  +-----------------------------------+
                  |      FoodLoop .NET Backend        |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+-----------------------+
            | REST (POST /monitoring/analyze)               | REST (POST /pricing/recommend)| REST (POST /pricing/knowledge/ingest)
            v                                               v                       v
+-----------------------+                       +-----------------------+   +-----------------------+
| Inventory Monitoring  |                       |     Pricing Agent     |   | Historical Pricing    |
|         Agent         |                       |     Recommendation    |   | Ingestion Service     |
+-----------+-----------+                       +-----------+-----------+   +-----------+-----------+
            |                                               |                           |
            v                                               v                           v
+-----------------------+                       +-----------------------+   +-----------------------+
|  Context & Risk       |                       |  Deterministic        |   | Document Builder      |
|  Signals Assessment   |                       |  Pricing Signals      |   | (Bilingual Natural)   |
+-----------+-----------+                       +-----------+-----------+   +-----------+-----------+
            |                                               |                           |
            v                                               v                           v
+-----------------------+                       +-----------------------+   +-----------------------+
| Route: NO_ACTION or   |                       | Store-Aware Historical|   | Local BGE-M3 (1024-d) |
|        PRICING        |                       | Vector Retrieval      |   | Batch Embedding       |
+-----------------------+                       +-----------+-----------+   +-----------+-----------+
                                                            |                           |
                                                            v                           v
                                                +-----------------------+   +-----------------------+
                                                | LLM Structured        |   | Qdrant / VectorStore  |
                                                | Recommendation (0-15%)|   | Idempotent Upsert     |
                                                +-----------+-----------+   +-----------------------+
                                                            |
                                                            v
                                                +-----------------------+
                                                | Action Policy         |
                                                | Interpretation        |
                                                +-----------------------+
```

---

## ✨ Key Features

- **Multilingual Arabic + English Semantic Retrieval**: Local `BAAI/bge-m3` embedding model producing 1024-dimensional normalized dense vectors.
- **Store-Scoped Batch Pricing**: Process batch requests per store with strict product-level data isolation.
- **Deterministic Pricing Signals**: Python-calculated inventory coverage days, demand ratio, and expiry pressure prior to LLM reasoning.
- **Store Operating Modes**: Supported modes `ASSISTED` (maps to `APPROVAL_REQUIRED`) and `AUTONOMOUS` (maps to `AUTOMATIC_EXECUTION_ELIGIBLE`).
- **Historical Pricing Ingestion Pipeline**: Decoupled, idempotent ingestion endpoint (`POST /api/v1/pricing/knowledge/ingest`) converting authoritative .NET historical pricing events into vector-embedded factual knowledge documents.
- **Qdrant Production Vector Store**: Multi-index payload search (`store_id`, `product_id`, `category`), cosine distance, and strict vector dimension validation (1024-d).
- **Open-Meteo Weather Provider**: Integrated weather provider abstraction with support for mock and production Open-Meteo APIs.
- **Nager.Date Holiday Provider**: Integrated public/national holiday context provider powered by Nager.Date API (`v4`) for country-level occasion evidence (Egypt `EG` default).
- **Auditable Rationales**: Enforces concise business reasons without exposing chain-of-thought or raw reasoning traces.
- **Robust Failure Boundaries**: Zero silent fallbacks to 0% discounts or fake historical data on infrastructure failure. If `VECTOR_STORE_PROVIDER=qdrant` and Qdrant is unavailable, the service fails explicitly without silent fallback.

---

## 🛠️ Environment Configuration & Deployment Modes

### Environment Modes

1. **TEST Mode (`APP_ENV=test`)**: Uses `FakeEmbeddingProvider`, `InMemoryVectorStore`, mocked LLM, mocked weather, and mocked holidays. Fully offline.
2. **DEVELOPMENT Mode (`APP_ENV=development`)**: Uses local `LocalBGEEmbeddingProvider` (BAAI/bge-m3), `InMemoryVectorStore` or local Qdrant, Open-Meteo, Nager.Date.
3. **PRODUCTION Mode (`APP_ENV=production`)**: Uses real LLM provider, `LocalBGEEmbeddingProvider` (BAAI/bge-m3 1024-d), Qdrant Vector Database, real external context providers, strict settings validation.

### Configuration Reference

Set the following environment variables in `.env`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_NAME` | `"FoodLoop AI Service"` | Application display name |
| `APP_ENV` | `"development"` | Environment (`development`, `test`, `production`) |
| `APP_VERSION` | `"1.0.0"` | Application version string |
| `OPENAI_API_KEY` | `""` | OpenAI / LLM provider API key |
| `OPENAI_MODEL` | `"gpt-4o-mini"` | LLM model name |
| `OPENAI_BASE_URL` | `""` | Optional base URL for custom OpenAI-compatible endpoints |
| `OPENAI_TIMEOUT_SECONDS` | `30.0` | Timeout in seconds for LLM calls |
| `EMBEDDING_PROVIDER` | `"local_bge_m3"` | Embedding backend (`local_bge_m3`, `openai`, `fake`) |
| `EMBEDDING_MODEL` | `"BAAI/bge-m3"` | HuggingFace model identifier |
| `EMBEDDING_VECTOR_SIZE` | `1024` | Vector dimension size (must be 1024 for BGE-M3) |
| `EMBEDDING_DEVICE` | `"cpu"` | Device execution (`cpu` or `cuda`) |
| `VECTOR_STORE_PROVIDER` | `"memory"` | Vector store backend (`memory` or `qdrant`) |
| `QDRANT_URL` | `"http://localhost:6333"` | Qdrant database endpoint |
| `QDRANT_API_KEY` | `""` | Qdrant API key |
| `QDRANT_COLLECTION_NAME` | `"foodloop_pricing_knowledge_bge_m3"` | Production collection name |
| `QDRANT_VECTOR_SIZE` | `1024` | Qdrant collection vector size |
| `PRICING_RETRIEVAL_TOP_K` | `5` | Top K historical items retrieved per product |
| `MAX_PRICING_BATCH_SIZE` | `50` | Maximum products per pricing batch recommendation request |
| `HISTORICAL_INGESTION_MAX_BATCH_SIZE` | `100` | Maximum events per historical pricing ingestion batch request |
| `WEATHER_PROVIDER` | `"mock"` | Weather provider backend (`mock` or `open_meteo`) |
| `EVENTS_PROVIDER` | `"mock"` | Holiday/events provider backend (`mock` or `nager_date`) |
| `DEFAULT_COUNTRY_CODE` | `"EG"` | Default country code for holiday lookup (ISO 2-letter) |

---

## 📡 API Specification

### Health & Operations
- `GET /health` — Fast process liveness check (`200 OK`)
- `GET /ready` — Service configuration & dependency readiness check (`200 OK` or `503 Service Unavailable`)
- `GET /version` — Version metadata (`200 OK`)

### Core Endpoints
- `POST /api/v1/monitoring/analyze` — Evaluates product inventory risk and determines workflow route (`NO_ACTION` vs `PRICING`).
- `POST /api/v1/pricing/recommend` — Evaluates batch product contexts, deterministic signals, and retrieved historical evidence to recommend discount percentages (0–15%) and policy action requirements.
- `POST /api/v1/pricing/knowledge/ingest` — Ingests authoritative historical pricing events from the .NET backend, transforms them into knowledge documents, embeds them in batch using BGE-M3 (1024-d), and upserts them idempotently to Qdrant.

---

## 🚀 Production Smoke Check CLI

Run the production readiness smoke check script to verify configuration and dependency reachability:

```bash
python -m app.cli.smoke_check
```

The smoke check verifies:
1. Environment & settings validation
2. OpenAI LLM model construction
3. Local BGE-M3 multilingual embedding provider (1024-d)
4. Qdrant vector store connectivity & collection readiness
5. Weather provider configuration
6. Holiday provider configuration

Exits `0` on success and `1` on failure without altering any business data.

---

## 🐳 Containerization & Production Deployment

> [!IMPORTANT]
> **Worker & Memory Safety Recommendation**:
> Because `BAAI/bge-m3` is loaded in-process (~2 GB RAM per process), run **1 worker default** (`WORKERS=1`) per container instance to avoid multiplying process memory usage. Horizontal scaling should be performed at the container orchestration layer (e.g. Docker Swarm, Kubernetes).

Build and run using Docker:

```bash
# Build multi-stage hardened production Docker image
docker build -t foodloop-ai-service:latest .

# Run production container
docker run -d \
  --name foodloop-ai-service \
  -p 8000:8000 \
  -e APP_ENV="production" \
  -e OPENAI_API_KEY="your-api-key" \
  -e VECTOR_STORE_PROVIDER="qdrant" \
  -e QDRANT_URL="https://your-qdrant-cluster:6333" \
  -e QDRANT_API_KEY="your-qdrant-key" \
  -e WEATHER_PROVIDER="open_meteo" \
  -e EVENTS_PROVIDER="nager" \
  foodloop-ai-service:latest
```

---

## 🧪 Opt-in Integration Test Execution

Run the complete offline test suite (289 passed unit tests, 0 failures):

```bash
pytest
```

To run explicit opt-in external integration tests:

```bash
# 1. OpenAI Live Test
RUN_EXTERNAL_INTEGRATION_TESTS=true OPENAI_API_KEY="your-key" pytest tests/test_llm_live.py

# 2. Open-Meteo Weather Live Test
RUN_EXTERNAL_INTEGRATION_TESTS=true pytest tests/test_open_meteo_weather.py

# 3. Nager.Date Holiday Live Test
RUN_EXTERNAL_INTEGRATION_TESTS=true pytest tests/test_nager_holidays.py

# 4. Qdrant Live CRUD Test
RUN_EXTERNAL_INTEGRATION_TESTS=true VECTOR_STORE_PROVIDER=qdrant QDRANT_URL="http://localhost:6333" pytest tests/test_qdrant_live.py

# 5. Live Multilingual Arabic Document + English Query Ingestion & Retrieval Test
RUN_EXTERNAL_INTEGRATION_TESTS=true VECTOR_STORE_PROVIDER=qdrant QDRANT_URL="http://localhost:6333" pytest tests/test_live_historical_ingestion.py
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

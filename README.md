# FoodLoop AI Service 🥑⚡

The **FoodLoop AI Service** is a high-performance, standalone Python microservice designed to handle intelligent inventory monitoring, risk assessment, and dynamic pricing orchestration for the FoodLoop platform. It seamlessly integrates with the FoodLoop .NET backend over RESTful APIs.

---

## 🏗️ Architecture Overview

The microservice is built using modern Python standards and an agentic workflow powered by **LangGraph**, **LangChain**, **FastAPI**, and **Pydantic v2**.

```text
                  +-------------------------+
                  |  FoodLoop .NET Backend  |
                  +------------+------------+
                               |
                               | REST API (POST /api/v1/monitoring/analyze)
                               v
                  +-------------------------+
                  |   FastAPI Web Engine    |
                  +------------+------------+
                               |
                               v
            +-------------------------------------+
            |  Inventory Monitoring Agent Graph   |
            +-------------------------------------+
            | 1. analyze_context (LLM)            |
            | 2. optional tool retrieval (Weather)|
            |    optional tool retrieval (Events) |
            | 3. assess_risk (Python + LLM)       |
            | 4. determine_route (Deterministic)  |
            +------------------+------------------+
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
     Route: NO_ACTION                       Route: PRICING
(Product is at low risk)              (Handed over for pricing)
```

---

## ✨ Key Features

- **FastAPI Core**: Async REST API endpoints with auto-generated OpenAPI documentation and Pydantic v2 schema validation.
- **LangGraph Workflow Orchestration**: Controlled multi-step agent graphs with single-pass tool invocation and state integrity preservation.
- **Hybrid Risk Assessment**: Combines high-speed deterministic Python calculations for expiry, inventory, and demand pressure with LLM interpretation of external environmental signals.
- **External Context Tools**: Provider-decoupled tools for retrieving weather forecasts and local events context based on location coordinates and shelf-life windows.
- **Audit-Friendly & Chain-of-Thought Guardrails**: Prompts enforce auditable reason strings without exposing internal chain-of-thought or raw reasoning traces.
- **Robust Exception Propagation**: Graph nodes and tools raise explicit domain errors (`WeatherToolError`, `LocalEventsToolError`, `RiskAssessmentMissingError`, `StateInvalidError`) without swallowing failures.

---

## 🛠️ Technology Stack

- **Language**: Python 3.10+ (Python 3.12 compatible)
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/)
- **Agent Graph**: [LangGraph](https://github.com/langchain-ai/langgraph)
- **LLM Orchestration**: [LangChain Core](https://github.com/langchain-ai/langchain) & [LangChain OpenAI](https://github.com/langchain-ai/langchain)
- **Data Validation**: [Pydantic v2](https://docs.pydantic.dev/latest/)
- **Configuration**: [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- **Testing**: [pytest](https://docs.pytest.org/)

---

## 📂 Project Structure

```text
api-service/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── health.py           # GET /api/v1/health
│   │       └── monitoring.py       # POST /api/v1/monitoring/analyze
│   ├── config/
│   │   └── settings.py             # Environment configuration (pydantic-settings)
│   ├── schemas/
│   │   ├── context_analysis.py     # ContextAnalysisResult & AllowedContext
│   │   ├── monitoring.py           # MonitoringRequest & MonitoringResponse
│   │   └── risk_assessment.py      # RiskAssessmentResult
│   ├── llm/
│   │   ├── factory.py              # Centralized LLM factory
│   │   └── model.py                # Provider-decoupled LLM interface
│   ├── tools/
│   │   ├── weather.py              # Weather forecast retrieval tool & models
│   │   └── events.py               # Local events retrieval tool & models
│   └── agents/
│       └── monitoring/             # Inventory Monitoring Agent Package
│           ├── state.py            # MonitoringAgentState schema
│           ├── prompts.py          # LLM system & human prompt templates
│           ├── risk_signals.py     # Deterministic Python risk signal formulas
│           ├── nodes.py            # LangGraph workflow nodes
│           └── graph.py            # StateGraph compilation & conditional routing
├── tests/
│   ├── test_context_analysis.py    # Unit tests for context analysis node
│   ├── test_context_tools.py       # Unit tests for weather & events tools
│   ├── test_e2e_scenarios.py       # 13 End-to-End integration scenario tests
│   ├── test_health.py              # API health check tests
│   ├── test_llm_factory.py         # LLM factory unit tests
│   ├── test_monitoring_agent.py    # Agent graph workflow tests
│   ├── test_monitoring_api.py      # API endpoint integration tests
│   ├── test_monitoring_schema.py   # Pydantic schema validation tests
│   ├── test_risk_assessment.py     # Risk assessment node & schema tests
│   ├── test_risk_signals.py        # Boundary tests for deterministic risk formulas
│   └── test_routing.py             # Deterministic routing policy tests
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## ⚡ Quick Start

### 1. Prerequisites

- Python 3.10+ (Python 3.12 recommended)
- `pip` or `uv`

### 2. Environment Setup

Clone the repository and initialize virtual environment:

```bash
git clone https://github.com/FoodLoopPlatform/AI_Service.git
cd AI_Service

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Environment Variables

Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

`.env` configuration parameters:

```env
APP_NAME="FoodLoop AI Service"
APP_ENV="development"
APP_VERSION="0.1.0"

OPENAI_API_KEY="your-openai-api-key"
OPENAI_MODEL="gpt-4o-mini"
```

### 4. Running the Development Server

Start the FastAPI server with auto-reload:

```bash
uvicorn app.main:app --reload --port 8000
```

Access interactive API documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Specification

### 1. Health Check
`GET /api/v1/health`

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "app_name": "FoodLoop AI Service",
  "version": "0.1.0"
}
```

### 2. Analyze Inventory Risk & Route
`POST /api/v1/monitoring/analyze`

**Request Payload**:
```json
{
  "product": {
    "id": "prod-123",
    "name": "Organic Milk",
    "category": "Dairy"
  },
  "inventory": {
    "quantity": 15,
    "original_price": 5.0,
    "current_price": 4.5,
    "price_floor": 2.0
  },
  "demand": {
    "sales_velocity": 1.2,
    "historical_sales": {
      "average_daily_sales": 3.0,
      "weekday_average": 2.8,
      "weekend_average": 3.5
    }
  },
  "expiry": {
    "expires_at": "2026-08-16T12:00:00Z",
    "hours_remaining": 22.0
  },
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "store_id": "store-001"
  },
  "timestamp": "2026-08-14T20:00:00Z"
}
```

**Response Payload (`200 OK`)**:
```json
{
  "route": "PRICING",
  "risk_level": "HIGH",
  "reason": "Imminent expiry within 24 hours coupled with low near-term sales velocity.",
  "confidence": 0.94
}
```

---

## 🧪 Testing

The repository maintains **100% self-contained unit and integration test coverage** (74 passed tests) with zero external network dependencies (LLMs and external tools are fully mocked during testing).

Run the full pytest suite:

```bash
pytest
```

---

## 🛡️ Business Boundaries & Architectural Invariants

1. **No Pricing Calculations in Monitoring Agent**: The Monitoring Agent only assesses risk (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) and determines routing (`NO_ACTION` vs `PRICING`). It never calculates discounts, target prices, or price floor adjustments.
2. **Deterministic Routing**: Workflow routing after risk assessment uses pure Python logic with zero LLM calls.
3. **Validation First**: Request payloads are validated via Pydantic v2 before agent execution. LLMs are never used to recover missing required business data.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

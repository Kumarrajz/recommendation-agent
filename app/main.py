# main.py
import os
import logging
from typing import List, Optional, Dict, Any, Set, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.database_repository import find_by_exact_or_partial_name
from app.agents import OrchestratorAgent
from models.schemas import Product, PublicProduct, RecommendationRequest, RecommendationResponse

from google import genai

# ==========================
# Config & Client
# ==========================
GEMINI_API_KEY = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

GEMINI_MODEL = config.GEMINI_MODEL
GEMINI_FALLBACK_MODEL = config.GEMINI_FALLBACK_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Recommendation Agent API",
    version="1.0.0",
    description="AI-powered product recommendation agent using Gemini + configurable data source (JSON/MongoDB/Postgres).",
)

logger = logging.getLogger("recommendation_agent")
logging.basicConfig(level=logging.INFO)


# ==========================
# Helpers
# ==========================
def to_public_product(prod: Product) -> PublicProduct:
    return PublicProduct(
        id=prod.id,
        name=prod.name,
        category=prod.category,
        brand=prod.brand,
        model=prod.model,
    )


# ==========================
# Orchestrator instance
# ==========================
orchestrator = OrchestratorAgent(client, GEMINI_MODEL, GEMINI_FALLBACK_MODEL)


# ==========================
# API endpoints
# ==========================
@app.get("/health")
def health_check():
    """Health check endpoint to verify service status."""
    return {
        "status": "healthy",
        "data_source_postgres": config.USE_POSTGRES,
        "gemini_model": config.GEMINI_MODEL
    }


@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    # 1) find primary
    primary = find_by_exact_or_partial_name(request.item_name)
    if not primary:
        return RecommendationResponse(primary_item=None, recommendations=[])

    # 2) orchestrated recommendation
    recs = orchestrator.recommend(primary, limit=request.limit)

    rec_products: List[PublicProduct] = []
    for r in recs:
        prod_obj = r.get("product")
        if prod_obj:
            # Convert internal Product model to PublicProduct
            public_prod = to_public_product(prod_obj)
            rec_products.append(public_prod)

    return RecommendationResponse(
        primary_item=to_public_product(primary),
        recommendations=rec_products,
    )

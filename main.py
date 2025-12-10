import os
import json
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mongo_repository import get_all_products, find_by_exact_or_partial_name
import config

from google import genai
from google.genai import types
import google.genai.errors as genai_errors


# ==========================
# Config & Client
# ==========================

GEMINI_API_KEY = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

# Optional: allow overriding the model via env var or config file
GEMINI_MODEL = config.GEMINI_MODEL
GEMINI_FALLBACK_MODEL = config.GEMINI_FALLBACK_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Recommendation Agent API",
    version="1.0.0",
    description="AI-powered product recommendation agent using Gemini + configurable data source (JSON/MongoDB).",
)


# ==========================
# Data Models
# ==========================

class Product(BaseModel):
    id: int
    name: str
    category: str
    brand: Optional[str] = None
    model: Optional[str] = None
    attributes: dict = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class PublicProduct(BaseModel):
    id: int
    name: str
    category: str
    brand: Optional[str] = None
    model: Optional[str] = None


class RecommendationRequest(BaseModel):
    item_name: str
    limit: int = 5


class RecommendationResponse(BaseModel):
    primary_item: Optional[PublicProduct]
    recommendations: List[PublicProduct]


# Repository layer: implemented in `mongo_repository.py` (currently JSON-backed)


# ==========================
# Candidate builder
# ==========================

def build_candidates(primary: Product) -> List[Product]:
    """
    Generic complements:
    - For phones:
        * cases / screen guards with matching compatible_model & compatible_brand
        * chargers with same port_type and compatible_brand or 'universal'
    - Generic:
        * same brand or overlapping tags
    """
    products = get_all_products()
    candidates: List[Product] = []

    for p in products:
        if p.id == primary.id:
            continue

        # PHONE LOGIC
        if primary.category == "phone":
            model = primary.model
            brand = primary.brand
            port = primary.attributes.get("port_type")

            # cases / tempered glass
            compatible_model = p.attributes.get("compatible_model")
            compatible_brand = p.attributes.get("compatible_brand")
            is_case_or_glass = p.category in ("phone_case", "screen_guard")

            if (
                is_case_or_glass
                and compatible_model == model
                and compatible_brand == brand
            ):
                candidates.append(p)
                continue

            # chargers
            is_charger = p.category == "charger"
            charger_port = p.attributes.get("port_type")
            charger_brand_ok = (
                p.attributes.get("compatible_brand") == brand
                or p.attributes.get("compatible_brand") == "universal"
            )

            if is_charger and charger_port == port and charger_brand_ok:
                candidates.append(p)
                continue

        # GENERIC COMPLEMENTS (exclude phones to avoid recommending other phones)
        if p.category != "phone":
            same_brand = (p.brand is not None and p.brand == primary.brand)
            tags_overlap = bool(set(p.tags).intersection(set(primary.tags)))

            if same_brand or tags_overlap:
                candidates.append(p)
                continue

    return candidates


# ==========================
# Gemini selector
# ==========================

def ai_pick_recommendation_ids(
    primary: Product,
    candidates: List[Product],
    limit: int,
) -> List[int]:
    """
    Try to use Gemini to pick candidate IDs.
    If quota/any error happens, fall back to simple deterministic logic:
    - Just return the first `limit` candidates.
    """
    primary_dict = primary.model_dump()
    candidates_dict = [c.model_dump() for c in candidates]

    prompt = f"""
You are a product recommendation engine.

Primary product:
{json.dumps(primary_dict, ensure_ascii=False)}

Candidate products (the ONLY products you can choose from):
{json.dumps(candidates_dict, ensure_ascii=False)}

Your task:
- Select up to {limit} products from the candidate list that are relevant recommendations.
- Relevance can be accessories, complements, or similar items based on category, brand, model, attributes, tags, or use-case.
- Use ONLY product IDs that appear in the candidate list.
- Do NOT invent new IDs or products.

Return ONLY valid JSON in this exact format:
{{
  "primary_item_id": <number>,
  "recommendation_ids": [<number>, ...]
}}
"""

    # --- Try primary + fallback models; if anything fails, we use deterministic fallback ---
    models_to_try = [GEMINI_MODEL, GEMINI_FALLBACK_MODEL]
    response = None

    for model_name in models_to_try:
        if not model_name:
            continue
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                    max_output_tokens=256,
                ),
            )
            break  # success
        except genai_errors.ClientError:
            # On any Gemini error (quota, auth, etc.), just try next model
            response = None
            continue

    # If no model succeeded, non-AI fallback: simply take first N candidate IDs
    if response is None:
        return [c.id for c in candidates[:limit]]

    # Parse model response
    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError:
        # Bad JSON from model → fallback
        return [c.id for c in candidates[:limit]]

    rec_ids = parsed.get("recommendation_ids", [])
    if not isinstance(rec_ids, list):
        return [c.id for c in candidates[:limit]]

    candidate_ids = {c.id for c in candidates}
    clean_ids: List[int] = []
    for rid in rec_ids:
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            continue
        if rid_int in candidate_ids and rid_int not in clean_ids:
            clean_ids.append(rid_int)

    # Apply limit & if model returned nothing, still fall back
    if not clean_ids:
        return [c.id for c in candidates[:limit]]
    return clean_ids[:limit]


# ==========================
# Utility
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
# API endpoints
# ==========================

@app.get("/health")
def health_check():
    """Health check endpoint to verify service status."""
    return {
        "status": "healthy",
        "data_source": "mongodb" if config.USE_MONGO else "json",
        "gemini_model": config.GEMINI_MODEL
    }

@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    # 1) find primary
    primary = find_by_exact_or_partial_name(request.item_name)
    if not primary:
        return RecommendationResponse(primary_item=None, recommendations=[])

    # 2) build candidates
    candidates = build_candidates(primary)
    if not candidates:
        return RecommendationResponse(
            primary_item=to_public_product(primary),
            recommendations=[],
        )

    # 3) use Gemini to pick IDs
    rec_ids = ai_pick_recommendation_ids(primary, candidates, request.limit)

    id_to_product = {p.id: p for p in candidates}
    rec_products: List[PublicProduct] = []
    for rid in rec_ids:
        prod = id_to_product.get(rid)
        if prod:
            rec_products.append(to_public_product(prod))

    return RecommendationResponse(
        primary_item=to_public_product(primary),
        recommendations=rec_products,
    )

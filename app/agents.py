import logging
import json
from typing import List, Optional, Any, Tuple, Dict
from google import genai
from google.genai import types
import google.genai.errors as genai_errors

from models.schemas import Product
from app.database_repository import get_all_products, get_products_by_category, get_products_by_brand, get_products_by_text_search
from app.domain_rules import (
    COMPATIBILITY_MAP, 
    PLACEHOLDER_BRANDS, 
    is_meaningful_brand, 
    tag_overlap_count, 
    attribute_match_score
)

logger = logging.getLogger("recommendation_agent")

class RetrieverAgent:
    """Responsible for fetching raw products from your data source."""
    def retrieve(self, primary: Product, max_results: int = 500) -> List[Product]:
        # Improved retrieval strategy: Use indexed queries instead of full scan
        candidates_map: Dict[int, Product] = {}

        # 1. Fetch same category (Essential)
        for p in get_products_by_category(primary.category, limit=max_results):
            candidates_map[p.id] = p

        # 2. Fetch compatible categories (Contextual)
        allowed_cats = COMPATIBILITY_MAP.get(primary.category, set())
        for cat in allowed_cats:
            if cat == primary.category:
                continue
            # We use a smaller limit for secondary categories to avoid noise
            for p in get_products_by_category(cat, limit=100): 
                candidates_map[p.id] = p

        # 3. Fetch same brand (Loyalty)
        if is_meaningful_brand(primary.brand):
            for p in get_products_by_brand(primary.brand, limit=200):
                candidates_map[p.id] = p
        else:
            # If brand is NOT meaningful (Generic, etc.), fetch "Universal" or "Generic" items
            # This ensures we get generic accessories for generic primaries
            for p in get_products_by_brand("Generic", limit=100):
                candidates_map[p.id] = p
            for p in get_products_by_brand("Universal", limit=100):
                candidates_map[p.id] = p

        # 4. Fetch by name/model text (Universal Discovery)
        # This finds "Case for iPhone 15" when primary is "iPhone 15"
        search_terms = []
        if primary.model and len(primary.model) >= 3:
            search_terms.append(primary.model)
        
        # Also try full name if it's distinguishable (e.g. "Galaxy S24")
        if primary.name:
            # simple heuristic: if name has digits, it might be specific enough
            if any(char.isdigit() for char in primary.name):
                search_terms.append(primary.name)

        for term in set(search_terms):
            for p in get_products_by_text_search(term, limit=100):
                candidates_map[p.id] = p

        # If we have very few results, fallback to a broader search or full scan (optional)
        # For now, if we have < 10 items, we might want to just get everything to be safe, 
        # or rely on the fact that if the DB is empty, we can't recommend anyway.
        if len(candidates_map) < 5:
             logger.debug("Low candidate count (%d), falling back to full scan for safety", len(candidates_map))
             all_prods = get_all_products(limit=1000)
             for p in all_prods:
                 candidates_map[p.id] = p

        candidates = list(candidates_map.values())
        
        # Sort/Prioritize based on original logic
        prioritized: List[Product] = []
        rest: List[Product] = []
        
        for p in candidates:
            if p.id == primary.id:
                continue
            
            # Strong preference for same-category items
            if p.category == primary.category:
                prioritized.append(p)
                continue
            
            # If primary has a meaningful brand, prioritize same-brand accessories
            if is_meaningful_brand(primary.brand) and is_meaningful_brand(p.brand) and p.brand == primary.brand:
                prioritized.append(p)
                continue
            
            # For sensitive categories, deprioritize unrelated consumer electronics
            if primary.category == "medical_equipment" and p.category in ("phone", "charger", "speaker", "phone_case", "screen_guard"):
                rest.append(p)
                continue
            
            rest.append(p)
            
        results = prioritized + rest
        return results[:max_results]


class CandidateBuilderAgent:
    """Prunes by category compatibility and applies conservative inclusion rules when no map exists."""
    def __init__(self, compat_map: Dict[str, Any] = COMPATIBILITY_MAP):
        self.compat_map = compat_map

    def build(self, primary: Product, raw_candidates: List[Product]) -> List[Product]:
        allowed = self.compat_map.get(primary.category)
        out: List[Product] = []
        
        # Pre-compute primary model/brand for fast checking
        p_model_lower = primary.model.lower() if primary.model else None
        p_brand_lower = primary.brand.lower() if primary.brand else None

        for p in raw_candidates:
            if p.id == primary.id:
                continue

            # 1. Global Bypass: Explicit Compatibility via Attributes
            # If a product explicitly says "I work with X", trust it 100%.
            explicit_attr_ok = False
            if p.attributes.get("compatible_model") == primary.model:
                explicit_attr_ok = True
            elif p.attributes.get("compatible_brand") == primary.brand and primary.brand:
                # Be careful: "Samsung" case compatible with "Samsung" phone is generic.
                # But if we are falling back, this might be okay.
                pass
            
            # 2. Global Bypass: Name-Based matching (The "Text Search" verification)
            # If Candidate Name contains Primary Model (e.g. "iPhone 15 Case" contains "iPhone 15")
            # AND categories are different (don't recommend phone for phone based on name match)
            name_match_ok = False
            if p_model_lower and p_model_lower in p.name.lower():
                if p.category != primary.category:
                    name_match_ok = True

            # If either global bypass is met, ADD it immediately (skip strict map checks)
            if explicit_attr_ok or name_match_ok:
                out.append(p)
                continue

            # 2b. Global Bypass: Universal/Generic Compatibility
            # If the candidate claims to be "Universal" or for "Generic" use, and isn't blocking the category.
            # E.g. a "Universal" power bank should be recommended for a "Generic" speaker.
            if p.attributes.get("compatible_brand") in ("Universal", "Generic", "Any"):
                # But still respect strict category incompatibility if defined (don't recommend universal car mount for a nebulizer?)
                # Actually, for "flipkart style", if it's universal, it's often shown.
                # Let's check tag overlap or category checks.
                # If primary is 'speaker', and candidate is 'charger' (universal), we want it even if map says no?
                # Map for speaker INCLUDES charger? No, it included aux_cable etc.
                # If map is present and EXCLUDES it, we might want to be careful.
                # But 'Generic' speaker + 'Universal' charger -> likely good match.
                pass
            
            # 3. Standard Compatibility Map Logic
            if allowed is not None:
                if p.category not in allowed:
                    continue

                # If candidate is cross-category, require explicit compatibility attributes or at least 2 tag overlaps
                if p.category != primary.category:
                    tag_overlap = tag_overlap_count(primary.tags, p.tags)
                    has_explicit_attr = bool(
                        p.attributes.get("compatible_model")
                        or p.attributes.get("compatible_brand")
                        or p.attributes.get("compatible_with")
                        or p.attributes.get(f"compatible_with_{primary.category}")
                        or p.attributes.get("compatible_with_speaker")
                        or p.attributes.get("compatible_with_medical_model")
                    )
                    if not has_explicit_attr and tag_overlap < 2:
                        continue
                out.append(p)
                continue

            # 4. Fallback (No Map Entry for Primary Category)
            # Be conservative
            tag_overlap = tag_overlap_count(primary.tags, p.tags)
            has_explicit_attr = bool(
                p.attributes.get("compatible_model")
                or p.attributes.get("compatible_brand")
                or p.attributes.get("compatible_with")
                or p.attributes.get(f"compatible_with_{primary.category}")
                or p.attributes.get("compatible_with_speaker")
                or p.attributes.get("compatible_with_medical_model")
            )
            if tag_overlap < 2 and not has_explicit_attr:
                continue

            # Example explicit exclusion: don't recommend phone screen guards for watch straps
            if primary.category == "watch_strap" and p.category == "screen_guard":
                continue

            out.append(p)
        return out


class ScorerAgent:
    """Deterministic scoring used for shortlisting and fallback."""
    def score(self, primary: Product, candidates: List[Product]) -> List[Product]:
        scored: List[Product] = []
        for c in candidates:
            score = 0
            score += attribute_match_score(primary, c)
            if is_meaningful_brand(primary.brand) and is_meaningful_brand(c.brand) and primary.brand == c.brand:
                score += 30
            tag_overlap = tag_overlap_count(primary.tags, c.tags)
            score += 10 * tag_overlap
            if c.category in COMPATIBILITY_MAP.get(primary.category, set()):
                score += 20
            if c.model_dump().get("image_url"):
                score += 2
            setattr(c, "_reco_score", score)
            setattr(c, "_reco_tag_overlap", tag_overlap)
            scored.append(c)
        scored.sort(key=lambda x: getattr(x, "_reco_score", 0), reverse=True)
        return scored


class LLMReRankerAgent:
    """Uses Gemini to re-rank a provided candidate shortlist."""
    def __init__(self, client: genai.Client, primary_model: Optional[str], fallback_model: Optional[str]):
        self.client = client
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    def rerank(self, primary: Product, candidates: List[Product], limit: int) -> Tuple[Optional[List[int]], Dict[str, str]]:
        if not self.primary_model and not self.fallback_model:
            return None, {}

        primary_dict = primary.model_dump()
        candidates_dict = [c.model_dump() for c in candidates]
        allowed = COMPATIBILITY_MAP.get(primary.category)
        allowed_list = sorted(list(allowed)) if allowed else []

        prompt = f"""
You are a product recommendation re-ranker.

Primary product:
{json.dumps(primary_dict, ensure_ascii=False)}

Candidate products (THE ONLY PRODUCTS YOU MAY CHOOSE FROM):
{json.dumps(candidates_dict, ensure_ascii=False)}

Allowed candidate categories for this primary:
{json.dumps(allowed_list, ensure_ascii=False)}

Important:
- You MUST choose only from the candidate products provided above.
- You MUST NOT select product categories that are not in the allowed list for this primary.
- If there is no allowed list for this primary, only select candidates that have explicit compatibility attributes (compatible_model, compatible_brand, compatible_with_<category>, compatible_with_speaker, compatible_with_medical_model, etc).
- Return ONLY valid JSON in the exact format described below.

Return format (JSON ONLY):
{{
  "primary_item_id": <number>,
  "recommendation_ids": [<number>, ...],   # up to {limit}
  "reasons": {{ "<id>": "<short reason>", ... }}   # optional
}}
"""
        models_to_try = [self.primary_model, self.fallback_model]
        response = None
        for model_name in models_to_try:
            if not model_name:
                continue
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                        max_output_tokens=512,
                    ),
                )
                break
            except genai_errors.ClientError as exc:
                logger.warning("Gemini client model %s failed: %s", model_name, exc)
                response = None
                continue

        if response is None:
            logger.info("LLM re-rank: no model responded")
            return None, {}

        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError:
            logger.warning("Gemini returned invalid JSON: %s", getattr(response, "text", None))
            return None, {}

        rec_ids = parsed.get("recommendation_ids", [])
        if not isinstance(rec_ids, list):
            logger.warning("LLM returned bad recommendation_ids type")
            return None, {}

        candidate_ids = {c.id for c in candidates}
        clean_ids: List[int] = []
        for rid in rec_ids:
            try:
                rid_int = int(rid)
            except (TypeError, ValueError):
                continue
            if rid_int in candidate_ids and rid_int not in clean_ids:
                clean_ids.append(rid_int)
            else:
                logger.debug("LLM recommended id %s not in candidate set or duplicate", rid)
        if not clean_ids:
            return None, {}

        reasons = parsed.get("reasons", {}) if isinstance(parsed.get("reasons", {}), dict) else {}
        return clean_ids[:limit], reasons


class ValidatorAgent:
    """Enforces business rules and safety checks on the final picks."""
    def validate(self, primary: Product, rec_ids: List[int], candidates: List[Product]) -> List[int]:
        id_to_c = {c.id: c for c in candidates}
        valid: List[int] = []
        allowed = COMPATIBILITY_MAP.get(primary.category)
        for rid in rec_ids:
            c = id_to_c.get(rid)
            if not c:
                continue

            # If allowed map exists, candidate category must be in it
            if allowed is not None and c.category not in allowed:
                logger.info("Validator blocked candidate %s with category %s not allowed for primary %s", rid, c.category, primary.id)
                continue

            # Special: do not recommend phones for speakers unless explicit compatibility exists
            if primary.category == "speaker" and c.category == "phone":
                if not (c.attributes.get("compatible_with_speaker") or c.attributes.get("compatible_with") or c.attributes.get("cross_compatible_with_speaker")):
                    logger.info("Validator blocked phone %s for speaker primary %s", rid, primary.id)
                    continue

            # Special: for speakers, only allow chargers/power_banks when explicitly compatible
            if primary.category == "speaker" and c.category in ("charger", "power_bank"):
                explicit_ok = bool(
                    c.attributes.get("compatible_with_speaker")
                    or c.attributes.get("compatible_with")
                    or c.attributes.get("cross_compatible_with_speaker")
                )
                
                # NEW: Strict compatibility check for Generic/Universal items
                # If candidate claims "Universal" compatibility, we allow it IF technical specs match (or if specs are missing, we lean permissive)
                if c.attributes.get("compatible_brand") in ("Universal", "Generic"):
                    explicit_ok = True
                
                power_match = False
                speaker_req = primary.attributes.get("required_voltage") or primary.attributes.get("required_input")
                charger_out = c.attributes.get("output_voltage") or c.attributes.get("output")
                
                # Voltage match check
                if speaker_req and charger_out:
                    # simplistic check: "5V" == "5V"
                    if str(speaker_req) == str(charger_out):
                        power_match = True
                elif speaker_req and not charger_out:
                    # if charger output unknown but it says "Universal", usually 5V USB. Allow.
                    if c.attributes.get("compatible_brand") in ("Universal", "Generic"):
                         power_match = True

                if not (explicit_ok or power_match):
                    logger.info("Validator blocked charger/power_bank %s for speaker %s (no explicit compatibility)", rid, primary.id)
                    continue

            # Critical: for medical_equipment, block unrelated consumer-electronics unless explicit medical compatibility
            if primary.category == "medical_equipment":
                # permit only items in medical categories or those explicitly declaring medical compatibility
                if c.category not in ("medical_supplies", "medical_accessory", "filters", "mask", "tubing", "mouthpiece", "medical_equipment"):
                    # allow only if candidate explicitly claims compatibility with this medical device
                    explicit_med_ok = bool(
                        c.attributes.get("compatible_with_medical_model")
                        or c.attributes.get("compatible_brand") == primary.brand
                        or c.attributes.get("compatible_model") == primary.model
                    )
                    if not explicit_med_ok:
                        logger.info("Validator blocked non-medical candidate %s for medical primary %s", rid, primary.id)
                        continue

            # Printer consumable safety: do not recommend other-brand cartridges unless cross_compatible flag present
            if primary.category == "printer" and c.category in ("ink_cartridge", "toner"):
                comp_brand = c.attributes.get("compatible_brand")
                if comp_brand and comp_brand != primary.brand:
                    if not c.attributes.get("cross_compatible", False):
                        logger.info("Validator blocked incompatible cartridge %s for printer %s", rid, primary.id)
                        continue

            valid.append(rid)
        return valid


class OrchestratorAgent:
    def __init__(self, client: genai.Client, primary_model: Optional[str], fallback_model: Optional[str]):
        self.retriever = RetrieverAgent()
        self.builder = CandidateBuilderAgent()
        self.scorer = ScorerAgent()
        self.reranker = LLMReRankerAgent(client, primary_model, fallback_model)
        self.validator = ValidatorAgent()

    def recommend(self, primary: Product, limit: int = 5) -> List[Dict[str, Any]]:
        # 1) retrieve raw candidates
        raw_candidates = self.retriever.retrieve(primary, max_results=1000)
        logger.debug("Retrieved %d raw candidates for primary %s", len(raw_candidates), primary.id)

        # 2) build candidates (category-first pruning / conservative rules)
        candidates = self.builder.build(primary, raw_candidates)
        logger.info("CandidateBuilder produced %d candidates for primary %s (category=%s)", len(candidates), primary.id, primary.category)
        if not candidates:
            logger.info("No candidates after CandidateBuilder for primary %s", primary.id)
            return []

        # 3) deterministic score & short-list
        scored = self.scorer.score(primary, candidates)
        top_m_limit = 30
        top_m = scored[:top_m_limit]
        logger.debug("Top-%d candidates after scoring: %s", top_m_limit, [(c.id, getattr(c, "_reco_score", 0)) for c in top_m])

        # 4) try LLM re-rank
        rec_ids, reasons = self.reranker.rerank(primary, top_m, limit)

        # fallback deterministic if LLM failed
        if not rec_ids:
            logger.info("LLM failed or returned nothing — falling back to deterministic top-N")
            rec_ids = [c.id for c in top_m[:limit]]
            reasons = {}

        # 5) validate
        valid_ids = self.validator.validate(primary, rec_ids, top_m)

        # If validator filtered everything, fallback again to deterministic top-n (but only same-category)
        if not valid_ids:
            logger.info("Validator filtered all picks; falling back to deterministic top-N limited to same-category")
            same_cat = [c.id for c in top_m if c.category == primary.category]
            valid_ids = same_cat[:limit] or [c.id for c in top_m[:limit]]

        # 6) Build ordered output
        id_to_p = {p.id: p for p in top_m}
        recommendations: List[Dict[str, Any]] = []
        for rid in valid_ids[:limit]:
            p = id_to_p.get(rid)
            if not p:
                p = next((s for s in scored if s.id == rid), None)
            if not p:
                continue
            
            # Helper to create public product here to keep agents independent of main.py's to_public logic
            # or we rely on main.py to convert. The Orchestrator returns dicts.
            # We'll just return the Product object in the dict, and main can convert.
            # But the original returned {product: PublicProduct...}.
            # Let's keep returning {product: PublicProduct} but we need to convert.
            # I can just import to_public_product if I move it to domain_rules or utils.
            # OR I just return the Product object and let main.py convert.
            # Return {product: Product, ...} is cleaner.
            recommendations.append({"product": p, "reason": reasons.get(str(rid)) or reasons.get(rid) or None})
        return recommendations

from typing import Dict, Set, List, Optional
from models.schemas import Product

PLACEHOLDER_BRANDS = {"generic", "unknown", "n/a", "", None}

# Conservative compatibility map. Add categories deliberately.
COMPATIBILITY_MAP: Dict[str, Set[str]] = {
    "phone": {"phone", "phone_case", "screen_guard", "charger", "pouch", "earbuds", "power_bank"},
    "watch_strap": {"watch_strap", "watch_tool", "spring_bar", "watch_box"},
    "charger": {"cable", "adapter", "power_bank"},
    "printer": {"ink_cartridge", "toner", "print_head", "maintenance_kit", "paper"},
    "tv": {"soundbar", "home_theatre", "remote", "wall_mount"},
    "pan": {"scrubber", "spatula", "lid", "pan_care_kit"},
    "nebulizer": {"mask", "filters", "tubing", "mouthpiece"},
    # VERY IMPORTANT: medical_equipment should only match medical accessories/consumables.
    "medical_equipment": {"medical_equipment", "medical_supplies", "medical_accessory", "filters", "mask", "tubing", "mouthpiece"},
    # Be conservative for speakers: prefer direct speaker accessories only (no chargers/power banks unless explicit)
    "speaker": {"speaker", "speaker_stand", "aux_cable", "bluetooth_transmitter", "case"},
    # extend with your categories...
}

def is_meaningful_brand(brand: Optional[str]) -> bool:
    if brand is None:
        return False
    return str(brand).strip().lower() not in PLACEHOLDER_BRANDS

def tag_overlap_count(a_tags: Optional[List[str]], b_tags: Optional[List[str]]) -> int:
    if not a_tags or not b_tags:
        return 0
    return len(set(a_tags).intersection(set(b_tags)))

def attribute_match_score(primary: Product, candidate: Product) -> int:
    """Domain-specific attribute scoring. Extend for new categories as needed."""
    score = 0
    # Phone strong match
    if primary.category == "phone":
        if candidate.category in ("phone_case", "screen_guard", "pouch", "charger"):
            if candidate.attributes.get("compatible_model") and candidate.attributes.get("compatible_brand"):
                if (candidate.attributes.get("compatible_model") == primary.model
                        and candidate.attributes.get("compatible_brand") == primary.brand):
                    score += 50
            if candidate.category == "charger":
                if candidate.attributes.get("port_type") == primary.attributes.get("port_type"):
                    score += 25

    # Watch strap
    if primary.category == "watch_strap":
        if candidate.attributes.get("size_mm") and primary.attributes.get("size_mm"):
            if candidate.attributes.get("size_mm") == primary.attributes.get("size_mm"):
                score += 40

    # Printer series/brand
    if primary.category == "printer":
        if candidate.category in ("ink_cartridge", "toner", "print_head"):
            if (candidate.attributes.get("compatible_series") and primary.attributes.get("series")
               and candidate.attributes.get("compatible_series") == primary.attributes.get("series")):
                score += 50
            if candidate.attributes.get("compatible_brand") and candidate.attributes.get("compatible_brand") == primary.brand:
                score += 20

    # Nebulizer compatibility
    if primary.category == "nebulizer":
        if candidate.category in ("mask", "filters", "tubing"):
            if candidate.attributes.get("compatible_model") == primary.model or candidate.attributes.get("compatible_brand") == primary.brand:
                score += 40

    # Speaker: prefer accessories explicitly tagged as speaker accessories or with explicit compatibility
    if primary.category == "speaker":
        if candidate.attributes.get("compatible_with_speaker") or ("speaker" in (candidate.tags or [])):
            score += 20

    # Medical equipment: high score for explicit medical compatibility fields
    if primary.category == "medical_equipment":
        if candidate.category in ("medical_supplies", "medical_accessory", "filters", "mask", "tubing", "mouthpiece"):
            if candidate.attributes.get("compatible_model") and candidate.attributes.get("compatible_brand"):
                if (candidate.attributes.get("compatible_model") == primary.model
                        or candidate.attributes.get("compatible_brand") == primary.brand):
                    score += 80
            # consumables that explicitly list the compatible device or brand
            if candidate.attributes.get("compatible_with_medical_model") == primary.model:
                score += 60
            # matching tags
            if tag_overlap_count(primary.tags, candidate.tags) > 0:
                score += 10

    return score

"""Repository layer prepared for MongoDB but currently backed by local JSON.

This module exposes small helper functions used by `main.py` to fetch
product data. By default the implementation reads `products.json` from the
project root and returns Pydantic `Product` instances. Example MongoDB
integration code is left commented below for future use.

Important: to avoid circular imports we import the `Product` model inside the
functions at runtime rather than at module import time.
"""

from pathlib import Path
import json
from typing import List, Optional

# Import config values
from config import PRODUCTS_PATH, USE_MONGO, MONGO_URI, MONGO_DB, MONGO_COLLECTION

# Import MongoDB dependencies only if needed
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

DATA_FILE = PRODUCTS_PATH


def _load_json_data() -> List[dict]:
	"""Load product data from JSON file."""
	try:
		with open(DATA_FILE, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return []


def _load_mongo_data() -> List[dict]:
	"""Load product data from MongoDB collection."""
	if not MONGO_AVAILABLE:
		print("MongoDB not available. Install pymongo to enable MongoDB support.")
		return []

	try:
		client = MongoClient(MONGO_URI)
		db = client[MONGO_DB]
		collection = db[MONGO_COLLECTION]
		documents = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB _id field
		client.close()
		return documents
	except ConnectionFailure:
		print(f"Failed to connect to MongoDB at {MONGO_URI}")
		return []
	except Exception as e:
		print(f"Error loading data from MongoDB: {e}")
		return []


def get_all_products() -> List[object]:
	"""Return a list of `Product` instances loaded from the configured data source.

	The return type is `List[object]` to avoid importing main's Product at
	module import time; callers can treat the returned objects as the
	appropriate Pydantic model.
	"""
	if USE_MONGO:
		data = _load_mongo_data()
	else:
		data = _load_json_data()

	if not data:
		return []

	# import locally to avoid circular import
	from main import Product
	return [Product.parse_obj(item) for item in data]

def find_by_exact_or_partial_name(name: str) -> Optional[object]:
	"""Find a product by exact or partial name using the configured data source.

	Returns a `Product` instance or `None`.
	"""
	lowered = name.strip().lower()
	products = get_all_products()

	# exact match
	for p in products:
		if p.name.lower() == lowered:
			return p

	# word-based partial match: all words in the search must appear in the product name
	words = lowered.split()
	for p in products:
		if all(word in p.name.lower() for word in words):
			return p

	return None
# --------------------
# MongoDB integration notes:
# - Set USE_MONGO=true in config.json or environment variable to enable MongoDB
# - Ensure MongoDB is running and accessible at MONGO_URI
# - Install pymongo: pip install pymongo
# - The collection should contain documents matching the Product schema
# --------------------

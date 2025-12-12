"""
Repository layer for products with robust Postgres support (connection pool),
safe parameterized queries, JSON/array handling, and fallback to local JSON.

Config values required in app.config:
  - PRODUCTS_PATH: path to local JSON fallback
  - USE_POSTGRES: bool (enable Postgres usage)
  - POSTGRES_DSN: e.g. "postgresql://user:pass@host:5432/dbname"
"""

from pathlib import Path
import json
import logging
from typing import List, Optional, Any, Iterable, Dict
import threading

from app.config import PRODUCTS_PATH, USE_POSTGRES, POSTGRES_DSN

# psycopg2 optional import
try:
    import psycopg2
    import psycopg2.extras
    from psycopg2 import pool
    PSYCOPG2_AVAILABLE = True
except Exception:
    psycopg2 = None
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DATA_FILE = Path(PRODUCTS_PATH)
# connection pool will be created lazily and be module-global
_pool_lock = threading.Lock()
_conn_pool: Optional[pool.ThreadedConnectionPool] = None


def _ensure_pool(minconn: int = 1, maxconn: int = 10):
    """Initialize global pool if not already created. Thread-safe."""
    global _conn_pool
    if not PSYCOPG2_AVAILABLE:
        return None
    if not POSTGRES_DSN:
        return None
    if _conn_pool is None:
        with _pool_lock:
            if _conn_pool is None:
                try:
                    _conn_pool = pool.ThreadedConnectionPool(minconn, maxconn, dsn=POSTGRES_DSN)
                    logger.info("Postgres connection pool created (min=%s max=%s)", minconn, maxconn)
                except Exception as e:
                    logger.exception("Failed to create Postgres connection pool: %s", e)
                    _conn_pool = None
    return _conn_pool


def _get_conn():
    """Get a connection from the pool (or None if unavailable). Use with finally to put back."""
    p = _ensure_pool()
    if not p:
        return None
    try:
        return p.getconn()
    except Exception:
        logger.exception("Failed to fetch connection from pool.")
        return None


def _put_conn(conn):
    """Return connection to pool (no-op if pool missing)."""
    p = _conn_pool
    if p and conn:
        try:
            p.putconn(conn)
        except Exception:
            logger.exception("Failed to put connection back to pool.")


def _load_json_data() -> List[dict]:
    """Load product data from JSON file. Returns empty list on error."""
    if not DATA_FILE.exists():
        logger.warning("Products JSON file not found at %s", DATA_FILE)
        return []

    try:
        with DATA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning("Products JSON root is not a list; got %s", type(data).__name__)
            return []
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in products file %s: %s", DATA_FILE, e)
        return []
    except Exception as e:
        logger.exception("Unexpected error loading products JSON: %s", e)
        return []


def _rows_to_dicts(rows) -> List[dict]:
    """Convert RealDict rows to plain dicts."""
    return [dict(r) for r in rows] if rows else []


# ----------------------
# Postgres helper queries
# ----------------------
def _load_postgres_data(limit: Optional[int] = None, offset: int = 0) -> List[dict]:
    """
    Query Postgres for product rows. Returns list of dicts.
    Uses connection pool if available. Safe parameterized queries.
    """
    if not (PSYCOPG2_AVAILABLE and POSTGRES_DSN):
        logger.debug("Postgres unavailable or DSN missing.")
        return []

    conn = _get_conn()
    if not conn:
        return []

    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sql = """
            SELECT id,
                   name,
                   category,
                   brand,
                   model,
                   attributes::jsonb AS attributes,
                   CASE WHEN tags IS NULL THEN NULL ELSE array_to_json(tags) END AS tags,
                   image_url,
                   description
            FROM products
            ORDER BY id
        """
        params: List[Any] = []
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))
        if offset:
            sql += " OFFSET %s"
            params.append(int(offset))
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return _rows_to_dicts(rows)
    except Exception as e:
        logger.exception("Failed to load products from Postgres: %s", e)
        return []
    finally:
        _put_conn(conn)


def _get_postgres_product_by_id(product_id: Any) -> Optional[dict]:
    """Fetch single product dict by id (Postgres)."""
    if not (PSYCOPG2_AVAILABLE and POSTGRES_DSN):
        return None
    conn = _get_conn()
    if not conn:
        return None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sql = """
            SELECT id, name, category, brand, model,
                   attributes::jsonb AS attributes,
                   CASE WHEN tags IS NULL THEN NULL ELSE array_to_json(tags) END AS tags,
                   image_url, description
            FROM products
            WHERE id = %s
            LIMIT 1
        """
        cur.execute(sql, (product_id,))
        row = cur.fetchone()
        cur.close()
        return dict(row) if row else None
    except Exception:
        logger.exception("Postgres query failed for id=%r", product_id)
        return None
    finally:
        _put_conn(conn)


def _search_postgres_by_name(name: str) -> Optional[dict]:
    """
    Try exact then simple partial (word AND) search in Postgres.
    Returns first matching row or None.
    """
    if not (PSYCOPG2_AVAILABLE and POSTGRES_DSN):
        return None
    lowered = name.strip().lower()
    if not lowered:
        return None

    conn = _get_conn()
    if not conn:
        return None
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # exact match
        cur.execute("SELECT * FROM products WHERE lower(name) = %s LIMIT 1;", (lowered,))
        row = cur.fetchone()
        if row:
            cur.close()
            return dict(row)
        # partial word-based AND (each word must be present)
        words = [w for w in lowered.split() if w]
        if words:
            conditions = " AND ".join("name ILIKE %s" for _ in words)
            params = [f"%{w}%" for w in words]
            sql = f"SELECT * FROM products WHERE {conditions} LIMIT 1;"
            cur.execute(sql, params)
            row = cur.fetchone()
            cur.close()
            if row:
                return dict(row)
        cur.close()
    except Exception:
        logger.exception("Postgres name search failed for %r", name)
    finally:
        _put_conn(conn)
    return None


def _search_postgres_by_tag(tag: str) -> List[dict]:
    """Return products that have the tag in their tags[] column (case-insensitive)."""
    if not (PSYCOPG2_AVAILABLE and POSTGRES_DSN):
        return []
    tag = tag.strip().lower()
    if not tag:
        return []
    conn = _get_conn()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sql = """
            SELECT id, name, category, brand, model,
                   attributes::jsonb AS attributes,
                   CASE WHEN tags IS NULL THEN NULL ELSE array_to_json(tags) END AS tags,
                   image_url, description
            FROM products
            WHERE EXISTS (
              SELECT 1 FROM unnest(tags) AS t WHERE lower(t) = %s
            )
            ORDER BY id
        """
        cur.execute(sql, (tag,))
        rows = cur.fetchall()
        cur.close()
        return _rows_to_dicts(rows)
    except Exception:
        logger.exception("Postgres tag query failed for tag=%r", tag)
        return []
    finally:
        _put_conn(conn)


# -------------------------
# New Optimized Queries
# -------------------------
def _get_postgres_products_by_category(category: str, limit: int = 500) -> List[dict]:
    """Fetch products by category (Postgres)."""
    if not (PSYCOPG2_AVAILABLE and POSTGRES_DSN):
        return []
    conn = _get_conn()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sql = """
            SELECT id, name, category, brand, model,
                   attributes::jsonb AS attributes,
                   CASE WHEN tags IS NULL THEN NULL ELSE array_to_json(tags) END AS tags,
                   image_url, description
            FROM products
            WHERE category = %s
            LIMIT %s
        """
        cur.execute(sql, (category, limit))
        rows = cur.fetchall()
        cur.close()
        return _rows_to_dicts(rows)
    except Exception:
        logger.exception("Postgres category query failed for category=%r", category)
        return []
    finally:
        _put_conn(conn)


def _get_postgres_products_by_brand(brand: str, limit: int = 500) -> List[dict]:
    """Fetch products by brand (Postgres)."""
    if not (PSYCOPG2_AVAILABLE and POSTGRES_DSN):
        return []
    conn = _get_conn()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sql = """
            SELECT id, name, category, brand, model,
                   attributes::jsonb AS attributes,
                   CASE WHEN tags IS NULL THEN NULL ELSE array_to_json(tags) END AS tags,
                   image_url, description
            FROM products
            WHERE brand = %s
            LIMIT %s
        """
        cur.execute(sql, (brand, limit))
        rows = cur.fetchall()
        cur.close()
        return _rows_to_dicts(rows)
    except Exception:
        logger.exception("Postgres brand query failed for brand=%r", brand)
        return []
    finally:
        _put_conn(conn)
def _to_products(data: Iterable[dict]) -> List[object]:
    """
    Convert dicts to Pydantic Product instances (imported at runtime).
    If Product schema import fails, returns raw dicts.
    """
    if not data:
        return []
    try:
        # attempt import paths commonly used in projects
        from models.schemas import Product  # type: ignore
    except Exception:
        try:
            from app.models.schemas import Product  # type: ignore
        except Exception as e:
            logger.exception("Failed to import Product schema: %s", e)
            return [item for item in data]

    products = []
    for item in data:
        try:
            products.append(Product.parse_obj(item))
        except Exception as e:
            logger.warning("Failed to parse product item into Product model: %s; item=%r", e, item)
            # skip invalid items
    return products


def _get_postgres_products_by_text(query: str, limit: int = 500) -> List[dict]:
    """Fetch products where name ILIKE sequence (Postgres)."""
    if not (PSYCOPG2_AVAILABLE and POSTGRES_DSN):
        return []
    conn = _get_conn()
    if not conn:
        return []
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Search name and tags
        sql = """
            SELECT id, name, category, brand, model,
                   attributes::jsonb AS attributes,
                   CASE WHEN tags IS NULL THEN NULL ELSE array_to_json(tags) END AS tags,
                   image_url, description
            FROM products
            WHERE name ILIKE %s
            OR EXISTS (SELECT 1 FROM unnest(tags) t WHERE t ILIKE %s)
            LIMIT %s
        """
        wild = f"%{query}%"
        cur.execute(sql, (wild, wild, limit))
        rows = cur.fetchall()
        cur.close()
        return _rows_to_dicts(rows)
    except Exception:
        logger.exception("Postgres text query failed for query=%r", query)
        return []
    finally:
        _put_conn(conn)


# -------------------------
# Public repository methods
# -------------------------
def _load_data(limit: Optional[int] = None, offset: int = 0) -> List[dict]:
    """Load from Postgres if enabled, else JSON fallback (or if Postgres fails)."""
    if USE_POSTGRES:
        pg = _load_postgres_data(limit=limit, offset=offset)
        if pg:
            return pg
        logger.info("Falling back to local JSON because Postgres returned no rows or failed.")
    # return _load_json_data()


def get_all_products(limit: Optional[int] = None, offset: int = 0) -> List[object]:
    raw = _load_data(limit=limit, offset=offset)
    return _to_products(raw)


def get_product_by_id(product_id: Any) -> Optional[object]:
    """Try Postgres direct lookup, otherwise fallback to scanning JSON/models in memory."""
    if USE_POSTGRES and PSYCOPG2_AVAILABLE and POSTGRES_DSN:
        row = _get_postgres_product_by_id(product_id)
        if row:
            prods = _to_products([row])
            return prods[0] if prods else None
        # if Postgres query failed or returned nothing, continue to fallback

    # fallback: scan JSON-loaded data
    products = get_all_products()
    for p in products:
        try:
            pid = getattr(p, "id", None)
        except Exception:
            pid = p.get("id") if isinstance(p, dict) else None
        if pid == product_id or str(pid) == str(product_id):
            return p
    return None


def find_by_exact_or_partial_name(name: str) -> Optional[object]:
    """Find product by exact or partial name. Prefer Postgres server-side search."""
    lowered = (name or "").strip()
    if not lowered:
        return None

    if USE_POSTGRES and PSYCOPG2_AVAILABLE and POSTGRES_DSN:
        row = _search_postgres_by_name(lowered)
        if row:
            prods = _to_products([row])
            return prods[0] if prods else None

    # fallback to in-memory search
    lowered = lowered.lower()
    products = get_all_products()
    # exact
    for p in products:
        try:
            pname = p.name
        except Exception:
            pname = p.get("name") if isinstance(p, dict) else None
        if pname and pname.lower() == lowered:
            return p
    # partial words
    words = lowered.split()
    for p in products:
        try:
            pname = p.name.lower()
        except Exception:
            pname = (p.get("name") or "").lower() if isinstance(p, dict) else ""
        if all(word in pname for word in words):
            return p
    return None


def find_by_tag(tag: str) -> List[object]:
    """Return products matching tag (case-insensitive)."""
    if not tag:
        return []
    if USE_POSTGRES and PSYCOPG2_AVAILABLE and POSTGRES_DSN:
        rows = _search_postgres_by_tag(tag)
        if rows:
            return _to_products(rows)

    # fallback
    lowered = tag.strip().lower()
    results = []
    products = get_all_products()
    for p in products:
        try:
            tags = getattr(p, "tags", None)
        except Exception:
            tags = p.get("tags") if isinstance(p, dict) else None
        if not tags:
            continue
        if isinstance(tags, str):
            tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
        else:
            tag_list = [str(t).lower() for t in tags]
        if lowered in tag_list:
            results.append(p)
    return results


def get_products_by_category(category: str, limit: int = 500) -> List[object]:
    """Return products matching category."""
    if USE_POSTGRES and PSYCOPG2_AVAILABLE and POSTGRES_DSN:
        rows = _get_postgres_products_by_category(category, limit)
        if rows:
            return _to_products(rows)
    
    # fallback
    results = []
    products = get_all_products()
    for p in products:
        try:
            p_cat = getattr(p, "category", "")
        except Exception:
            p_cat = p.get("category") if isinstance(p, dict) else ""
        if p_cat == category:
            results.append(p)
            if len(results) >= limit:
                break
    return results


def get_products_by_brand(brand: str, limit: int = 500) -> List[object]:
    """Return products matching brand."""
    if USE_POSTGRES and PSYCOPG2_AVAILABLE and POSTGRES_DSN:
        rows = _get_postgres_products_by_brand(brand, limit)
        if rows:
            return _to_products(rows)

    # fallback
    results = []
    products = get_all_products()
    for p in products:
        try:
            p_brand = getattr(p, "brand", "")
        except Exception:
            p_brand = p.get("brand") if isinstance(p, dict) else ""
        if p_brand == brand:
            results.append(p)
            if len(results) >= limit:
                break
    return results


def get_products_by_text_search(query: str, limit: int = 500) -> List[object]:
    """Return products matching text query in name or tags."""
    if not query or not query.strip():
        return []
    
    if USE_POSTGRES and PSYCOPG2_AVAILABLE and POSTGRES_DSN:
        rows = _get_postgres_products_by_text(query.strip(), limit)
        if rows:
            return _to_products(rows)

    # fallback
    results = []
    products = get_all_products()
    q = query.strip().lower()
    for p in products:
        try:
            pname = str(getattr(p, "name", "") or "")
            ptags = getattr(p, "tags", []) or []
        except Exception:
            pname = str(p.get("name", "") or "") if isinstance(p, dict) else ""
            ptags = p.get("tags", []) if isinstance(p, dict) else []
        
        # Check name
        if q in pname.lower():
            results.append(p)
            continue
            
        # Check tags
        if isinstance(ptags, list):
            for t in ptags:
                if q in str(t).lower():
                    results.append(p)
                    break
        
        if len(results) >= limit:
            break
    return results

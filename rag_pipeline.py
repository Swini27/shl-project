import json
import os
import re
import chromadb

CATALOG_FILE = os.path.join(os.path.dirname(__file__), "catalog.json")
# On Render, CHROMA_DB_DIR points to the mounted persistent disk (/data/chroma_db).
# Locally it falls back to ./chroma_db next to this file.
DB_DIR = os.environ.get("CHROMA_DB_DIR", os.path.join(os.path.dirname(__file__), "chroma_db"))


def _parse_duration_minutes(duration_raw: str) -> int:
    match = re.search(r'(\d+)', duration_raw or "")
    return int(match.group(1)) if match else 0


def _matches_filters(item: dict, state) -> bool:
    """Manually check if a catalog item satisfies active hard filters (used for BM25 candidates)."""
    max_dur = getattr(state, "max_duration_minutes", None)
    if max_dur and max_dur > 0:
        dur = _parse_duration_minutes(item.get("duration_raw", ""))
        # If duration is unknown (0) or exceeds the max, filter it out
        if dur == 0 or dur > max_dur:
            return False
    lang = getattr(state, "language_filter", None)
    if lang and lang not in item.get("languages", []):
        return False
    if getattr(state, "remote_only", False) and item.get("remote", "").lower() != "yes":
        return False
    if getattr(state, "adaptive_only", False) and item.get("adaptive", "").lower() != "yes":
        return False
    return True


class RAGCatalog:
    def __init__(self):
        # Optional: cross-encoder re-ranker
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("Cross-encoder re-ranker loaded successfully.")
        except ImportError:
            self.cross_encoder = None
            print("INFO: sentence-transformers not installed. Re-ranking disabled.")

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=DB_DIR)
        self.collection_name = "shl_assessments"

        with open(CATALOG_FILE, 'r') as f:
            self.raw_catalog = json.load(f)
            self.catalog_map = {str(item.get("entity_id", i)): item for i, item in enumerate(self.raw_catalog)}

        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            if self.collection.count() == 0:
                self._build_index()
        except Exception:
            self.collection = self.chroma_client.create_collection(self.collection_name)
            self._build_index()

        # Optional: BM25 keyword index for hybrid search
        try:
            from rank_bm25 import BM25Okapi
            self._build_bm25_index(BM25Okapi)
            print("BM25 keyword index built successfully.")
        except ImportError:
            self.bm25 = None
            self.bm25_ids = []
            print("INFO: rank-bm25 not installed. Hybrid search disabled. Run: pip install rank-bm25")

    def _build_index(self):
        print("Building ChromaDB RAG index...")
        documents, metadatas, ids = [], [], []

        for i, item in enumerate(self.raw_catalog):
            entity_id = str(item.get("entity_id", i))
            name = item.get("name", "")
            desc = item.get("description", "")
            job_levels = item.get("job_levels_raw", "")
            keys = ", ".join(item.get("keys", []))
            duration = item.get("duration", "")
            langs = ", ".join(item.get("languages", []))

            doc_text = f"Name: {name}\nDescription: {desc}\nRole Relevance: {job_levels}\nCategories: {keys}\nDuration: {duration}\nLanguages: {langs}"
            test_type = item.get("keys", ["Unknown"])[0] if item.get("keys") else "Unknown"
            duration_minutes = _parse_duration_minutes(item.get("duration_raw", ""))
            primary_language = item.get("languages", [""])[0] if item.get("languages") else ""

            meta = {
                "name": name, "test_type": test_type,
                "link": item.get("link", item.get("url", "")),
                "duration_minutes": duration_minutes,
                "language": primary_language,
                "remote": item.get("remote", ""),
                "adaptive": item.get("adaptive", ""),
            }
            documents.append(doc_text)
            metadatas.append(meta)
            ids.append(entity_id)

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print(f"ChromaDB Index built with {len(ids)} assessments.")

    def _build_bm25_index(self, BM25Okapi):
        """Build in-memory BM25 index from the same catalog for keyword scoring."""
        corpus = []
        self.bm25_ids = []
        for entity_id, item in self.catalog_map.items():
            tokens = " ".join([
                item.get("name", ""),
                item.get("description", ""),
                item.get("job_levels_raw", ""),
                " ".join(item.get("keys", [])),
                " ".join(item.get("languages", [])),
            ]).lower().split()
            corpus.append(tokens)
            self.bm25_ids.append(entity_id)
        self.bm25 = BM25Okapi(corpus)

    def _build_where_clause(self, state) -> dict:
        conditions = []
        max_dur = getattr(state, "max_duration_minutes", None)
        if max_dur and max_dur > 0:
            conditions.append({"duration_minutes": {"$lte": max_dur}})
        lang = getattr(state, "language_filter", None)
        if lang:
            conditions.append({"language": {"$eq": lang}})
        if getattr(state, "remote_only", False):
            conditions.append({"remote": {"$eq": "yes"}})
        if getattr(state, "adaptive_only", False):
            conditions.append({"adaptive": {"$eq": "yes"}})
        if not conditions:
            return {}
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def search(self, state, top_k: int = 10) -> list:
        """
        Hybrid search: vector similarity (ChromaDB) + BM25 keyword scoring.
        Scores are combined as: 0.6 * vector_sim + 0.4 * bm25_score.
        Final candidates are optionally re-ranked by a cross-encoder.
        """
        query_parts = []
        if getattr(state, "job_role", None): query_parts.append(f"Role: {state.job_role}")
        if getattr(state, "seniority", None): query_parts.append(f"Level: {state.seniority}")
        if getattr(state, "technical_stack", None): query_parts.append(f"Tech Stack: {', '.join(state.technical_stack)}")
        if getattr(state, "soft_skills", None): query_parts.append(f"Soft Skills: {', '.join(state.soft_skills)}")
        if getattr(state, "assessment_types", None): query_parts.append(f"Test Type: {', '.join(state.assessment_types)}")
        if getattr(state, "other_constraints", None): query_parts.append(f"Constraints: {', '.join(state.other_constraints)}")
        query_text = " ".join(query_parts) if query_parts else "general assessment"

        fetch_k = min(top_k * 3, self.collection.count())
        where_clause = self._build_where_clause(state)

        # --- Step 1: Vector search ---
        query_kwargs = {
            "query_texts": [query_text],
            "n_results": fetch_k,
            "include": ["documents", "distances"],
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        results = self.collection.query(**query_kwargs)

        vec_scores: dict = {}
        vec_texts: dict = {}
        if results['ids'] and results['ids'][0]:
            dists = results['distances'][0]
            max_dist = max(dists) if dists else 1.0
            for doc_id, dist, doc_text in zip(results['ids'][0], dists, results['documents'][0]):
                vec_scores[doc_id] = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
                vec_texts[doc_id] = doc_text

        # --- Step 2: BM25 keyword search ---
        bm25_scores: dict = {}
        if getattr(self, 'bm25', None) and self.bm25_ids:
            raw = self.bm25.get_scores(query_text.lower().split())
            top_idxs = sorted(range(len(raw)), key=lambda i: raw[i], reverse=True)[:fetch_k]
            max_bm25 = raw[top_idxs[0]] if top_idxs else 0.0
            if max_bm25 > 0:
                for i in top_idxs:
                    eid = self.bm25_ids[i]
                    if eid in self.catalog_map and _matches_filters(self.catalog_map[eid], state):
                        bm25_scores[eid] = raw[i] / max_bm25

        # --- Step 3: Merge and compute hybrid scores ---
        ALPHA = 0.6  # vector weight; 0.4 goes to BM25
        all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
        candidates = []
        for eid in all_ids:
            if eid not in self.catalog_map:
                continue
            if not _matches_filters(self.catalog_map[eid], state):
                continue
            hybrid = ALPHA * vec_scores.get(eid, 0.0) + (1 - ALPHA) * bm25_scores.get(eid, 0.0)
            doc_text = vec_texts.get(eid, "")
            if not doc_text:
                item = self.catalog_map[eid]
                doc_text = f"{item.get('name','')} {item.get('description','')}"
            candidates.append({"id": eid, "doc_text": doc_text, "score": hybrid})

        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:fetch_k]

        # --- Step 4: Cross-encoder re-ranking ---
        if self.cross_encoder and candidates:
            pairs = [[query_text, c["doc_text"]] for c in candidates]
            ce_scores = self.cross_encoder.predict(pairs)
            for c, s in zip(candidates, ce_scores):
                c["score"] = float(s)
            candidates.sort(key=lambda x: x["score"], reverse=True)

        final_results = []
        for c in candidates[:top_k]:
            # Cross-encoder logits (ms-marco): > -5.0 is generally relevant. Hybrid scores: > 0.1 is decent.
            threshold = -5.0 if self.cross_encoder else 0.1
            if c["score"] >= threshold:
                final_results.append(self.catalog_map[c["id"]])
                
        return final_results

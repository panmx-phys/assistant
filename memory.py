"""Persistent vector memory using ChromaDB + Ollama fact extraction.

Recall uses a human-brain-inspired model:
  - Ebbinghaus forgetting curve: memories decay over time
  - Significance slows decay (important memories last longer)
  - Access count strengthens memories (rehearsal effect)
  - Recency boosts recently stored/accessed memories
"""
from __future__ import annotations

import math
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb

import config
import llm

# File to track last declutter run date
_DECLUTTER_STAMP = Path(config.CHROMA_PATH) / ".last_declutter"


class Memory:
    """Persistent vector memory using ChromaDB + Ollama for fact extraction."""

    def __init__(self):
        self._client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self._collection = self._client.get_or_create_collection(config.COLLECTION_NAME)
        self._lock = threading.Lock()
        # Auto-backfill unscored memories in background
        threading.Thread(target=self._auto_backfill, daemon=True).start()
        # Auto-declutter in a separate process if not yet run today
        self._maybe_auto_declutter()

    # ── Collection safety helpers ──────────────────────────────────────────

    @staticmethod
    def _is_missing_collection_error(err: Exception) -> bool:
        """Return True when Chroma reports a missing collection handle."""
        return (
            err.__class__.__name__ == "NotFoundError"
            or "does not exist" in str(err).lower()
        )

    def _refresh_collection(self):
        """Rebind to the configured collection, creating it if missing."""
        self._collection = self._client.get_or_create_collection(config.COLLECTION_NAME)

    def _run_with_collection_retry(self, fn):
        """Run fn and auto-recover once from stale/missing collection handles."""
        try:
            return fn()
        except Exception as e:
            if self._is_missing_collection_error(e):
                self._refresh_collection()
                return fn()
            raise

    # ── Fact extraction (richer details) ──────────────────────────────────

    def _extract_facts(self, user_msg: str, assistant_msg: str) -> list[dict]:
        """Extract facts with significance, topic, and emotion from a conversation pair.

        Returns list of dicts: {fact, significance, topic, emotion}
        """
        prompt = (
            "Extract key facts, preferences, and personal details from this conversation.\n"
            "For each fact, provide:\n"
            "  - A significance score (1-5):\n"
            "    1 = trivial preference (e.g. likes dark mode)\n"
            "    2 = minor detail (e.g. drinks coffee)\n"
            "    3 = notable fact (e.g. works as a teacher)\n"
            "    4 = important life detail (e.g. just got married)\n"
            "    5 = deeply significant (e.g. lost a loved one, major life crisis)\n"
            "  - A topic tag (one word: personal, work, health, hobby, relationship, finance, goal, opinion, other)\n"
            "  - The user's emotional tone (one word: neutral, happy, sad, excited, anxious, frustrated, curious, grateful)\n\n"
            "Return each fact on its own line in this format:\n"
            "[score|topic|emotion] detailed fact\n\n"
            "Example:\n"
            "[3|work|neutral] User works as a software engineer at a startup in Austin\n"
            "[4|relationship|happy] User just got engaged to their partner of 3 years named Alex\n"
            "[2|hobby|excited] User recently started learning piano and practices daily\n\n"
            "Be detailed — include names, places, dates, and context when mentioned.\n"
            "Only return scored facts, nothing else.\n"
            'If there are no notable facts, return only "NONE".\n\n'
            f"User: {user_msg}\nAssistant: {assistant_msg}"
        )
        t0 = time.monotonic()
        error = None
        response_text = ""
        model_used = config.EXTRACTION_MODEL
        try:
            response_text = llm.call_ollama(config.EXTRACTION_MODEL, prompt)
        except Exception as e:
            # Fall back to Gemini when Ollama is not installed or unavailable
            client = config._gemini_clients.get(
                config.EXTRACTION_FALLBACK_API_KEY, config._gemini_client
            )
            if client is not None:
                try:
                    response_text = llm.call_gemini_api(
                        config.EXTRACTION_FALLBACK_MODEL,
                        prompt,
                        temperature=0.2,
                        max_tokens=1024,
                        client=client,
                    )
                    model_used = config.EXTRACTION_FALLBACK_MODEL
                    error = None
                except Exception as gemini_err:
                    error = str(gemini_err)
            else:
                error = str(e)
        elapsed = int((time.monotonic() - t0) * 1000)

        llm.debug_log.log(
            call_type="fact_extraction",
            model=model_used,
            prompt=prompt,
            response=response_text,
            elapsed_ms=elapsed,
            error=error,
        )

        if not response_text or response_text.upper() == "NONE":
            return []

        results: list[dict] = []
        for line in response_text.splitlines():
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue
            # Parse [score|topic|emotion] fact format
            m = re.match(r"\[(\d)\|(\w+)\|(\w+)\]\s*(.+)", line)
            if m:
                score = max(1, min(5, int(m.group(1))))
                topic = m.group(2).strip().lower()
                emotion = m.group(3).strip().lower()
                fact = m.group(4).strip()
            else:
                # Fallback: try old [score] format
                m2 = re.match(r"\[(\d)\]\s*(.+)", line)
                if m2:
                    score = max(1, min(5, int(m2.group(1))))
                    fact = m2.group(2).strip()
                else:
                    fact = line.lstrip("•-* 0123456789.)")
                    score = 2
                topic = "other"
                emotion = "neutral"
            if fact and len(fact) > 2:
                results.append({
                    "fact": fact,
                    "significance": score,
                    "topic": topic,
                    "emotion": emotion,
                })
        return results

    # ── Time helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _relative_time(iso_ts: str) -> str:
        """Convert an ISO timestamp to a human-friendly relative time string."""
        try:
            stored = datetime.fromisoformat(iso_ts)
            if stored.tzinfo is None:
                stored = stored.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - stored
            days = delta.days
            if days == 0:
                hours = delta.seconds // 3600
                if hours == 0:
                    return "just now"
                return f"{hours}h ago"
            if days == 1:
                return "yesterday"
            if days < 7:
                return f"{days} days ago"
            if days < 30:
                weeks = days // 7
                return f"{weeks} week{'s' if weeks > 1 else ''} ago"
            if days < 365:
                months = days // 30
                return f"{months} month{'s' if months > 1 else ''} ago"
            years = days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
        except Exception:
            return ""

    @staticmethod
    def _days_since(iso_ts: str) -> float:
        """Return fractional days since a timestamp, or 0 on error."""
        if not iso_ts:
            return 0.0
        try:
            stored = datetime.fromisoformat(iso_ts)
            if stored.tzinfo is None:
                stored = stored.replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - stored).total_seconds() / 86400)
        except Exception:
            return 0.0

    # ── Brain-inspired recall ─────────────────────────────────────────────

    @staticmethod
    def _memory_strength(days_old: float, significance: int, access_count: int) -> float:
        """Human-brain-inspired memory strength using Ebbinghaus forgetting curve.

        strength = (1 + rehearsal_bonus) * e^(-decay_rate * time)

        - Higher significance -> slower decay (important memories last longer)
        - More accesses -> stronger memory (rehearsal / spaced repetition)
        - Time naturally weakens all memories, but slowly for important ones
        """
        # Decay rate inversely proportional to significance (1-5)
        # sig=1 -> decay ~0.20/day (half-life ~3.5 days)
        # sig=3 -> decay ~0.05/day (half-life ~14 days)
        # sig=5 -> decay ~0.03/day (half-life ~25 days)
        decay_rate = 0.20 / (significance ** 1.2)
        # Rehearsal bonus: each access strengthens the memory (log scale)
        rehearsal_bonus = 0.15 * math.log1p(access_count)
        # Ebbinghaus-style exponential decay
        retention = math.exp(-decay_rate * days_old)
        return (1.0 + rehearsal_bonus) * retention

    def recall(self, query: str, limit: int = config.RECALL_LIMIT) -> list[tuple[str, str]]:
        """Recall memories ranked by similarity * brain-like memory strength.

        Combines:
          - Vector similarity (relevance to query)
          - Forgetting curve (time decay modulated by significance)
          - Rehearsal effect (access count strengthens memory)
        Returns list of (fact, relative_time).
        """
        try:
            def _query():
                with self._lock:
                    count = self._collection.count()
                    if count == 0:
                        return None
                    # Fetch more candidates for re-ranking
                    fetch_n = min(limit * 5, count)
                    return self._collection.query(
                        query_texts=[query],
                        n_results=fetch_n,
                        include=["documents", "metadatas", "distances"],
                    )

            results = self._run_with_collection_retry(_query)
            if results is None:
                return []
            if not results["documents"] or not results["documents"][0]:
                return []

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            scored = []
            for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
                sig = meta.get("significance", 2) if meta else 2
                ts = meta.get("stored_at", "") if meta else ""
                access_count = meta.get("access_count", 0) if meta else 0

                days_old = self._days_since(ts)
                similarity = 1.0 / (1.0 + dist)
                strength = self._memory_strength(days_old, sig, access_count)
                final_score = similarity * strength
                scored.append((doc_id, doc, final_score, ts))

            scored.sort(key=lambda x: x[2], reverse=True)
            top = scored[:limit]

            # Update access counts for recalled memories (rehearsal effect)
            self._bump_access([item[0] for item in top])

            return [(doc, self._relative_time(ts)) for _, doc, _, ts in top]
        except Exception:
            return []

    def _bump_access(self, ids: list[str]):
        """Increment access_count and update last_accessed for recalled memories."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            def _update():
                with self._lock:
                    for mem_id in ids:
                        result = self._collection.get(ids=[mem_id], include=["metadatas"])
                        if not result["metadatas"]:
                            continue
                        meta = dict(result["metadatas"][0])
                        meta["access_count"] = meta.get("access_count", 0) + 1
                        meta["last_accessed"] = now_iso
                        self._collection.update(ids=[mem_id], metadatas=[meta])

            self._run_with_collection_retry(_update)
        except Exception:
            pass

    # ── Store ─────────────────────────────────────────────────────────────

    def _store_sync(self, user_msg: str, assistant_msg: str):
        """Extract facts and store them with rich metadata (runs in background thread)."""
        extracted = self._extract_facts(user_msg, assistant_msg)
        ts = datetime.now(timezone.utc).isoformat()
        def _add():
            with self._lock:
                for item in extracted:
                    self._collection.add(
                        ids=[str(uuid.uuid4())],
                        documents=[item["fact"]],
                        metadatas=[{
                            "stored_at": ts,
                            "significance": item["significance"],
                            "topic": item.get("topic", "other"),
                            "emotion": item.get("emotion", "neutral"),
                            "access_count": 0,
                            "last_accessed": "",
                        }],
                    )

        self._run_with_collection_retry(_add)

    def store(self, user_msg: str, assistant_msg: str):
        """Fire-and-forget async fact extraction."""
        t = threading.Thread(target=self._store_sync, args=(user_msg, assistant_msg), daemon=True)
        t.start()

    def store_fact(self, fact: str, significance: int = 3):
        """Store a manually provided fact. Default significance=3 (user chose to remember)."""
        ts = datetime.now(timezone.utc).isoformat()
        def _add():
            with self._lock:
                self._collection.add(
                    ids=[str(uuid.uuid4())],
                    documents=[fact],
                    metadatas=[{
                        "stored_at": ts,
                        "significance": significance,
                        "topic": "other",
                        "emotion": "neutral",
                        "access_count": 0,
                        "last_accessed": "",
                    }],
                )

        self._run_with_collection_retry(_add)

    def get_all(self) -> list[str]:
        try:
            def _get():
                with self._lock:
                    if self._collection.count() == 0:
                        return []
                    return self._collection.get()["documents"]

            return self._run_with_collection_retry(_get)
        except Exception:
            return []

    # ── Scoring / backfill ────────────────────────────────────────────────

    def _score_fact(self, fact: str) -> int:
        """Use Ollama to assign a significance score (1-5) to an existing fact."""
        prompt = (
            "Rate this fact's emotional/personal significance from 1 to 5:\n"
            "  1 = trivial preference (e.g. likes dark mode)\n"
            "  2 = minor detail (e.g. drinks coffee)\n"
            "  3 = notable fact (e.g. works as a teacher)\n"
            "  4 = important life detail (e.g. just got married)\n"
            "  5 = deeply significant (e.g. lost a loved one)\n\n"
            f"Fact: {fact}\n\n"
            "Return only the number (1-5), nothing else."
        )
        try:
            resp = llm.call_ollama(config.EXTRACTION_MODEL, prompt, max_tokens=10)
            m = re.search(r"[1-5]", resp)
            return int(m.group()) if m else 2
        except Exception:
            return 2

    def backfill_significance(self) -> tuple[int, int]:
        """Score all memories that lack a significance value. Returns (scored, total)."""
        def _get_all_for_backfill():
            with self._lock:
                count = self._collection.count()
                if count == 0:
                    return 0, None
                return count, self._collection.get(include=["metadatas", "documents"])

        count, all_data = self._run_with_collection_retry(_get_all_for_backfill)
        if count == 0 or all_data is None:
            return 0, 0

        ids = all_data["ids"]
        docs = all_data["documents"]
        metas = all_data["metadatas"]

        unscored = [
            (id_, doc, meta)
            for id_, doc, meta in zip(ids, docs, metas)
            if not meta or meta.get("significance") is None
        ]

        for id_, doc, meta in unscored:
            score = self._score_fact(doc)
            updated_meta = dict(meta) if meta else {}
            updated_meta["significance"] = score
            def _update():
                with self._lock:
                    self._collection.update(ids=[id_], metadatas=[updated_meta])

            self._run_with_collection_retry(_update)

        return len(unscored), count

    def _auto_backfill(self):
        """Background backfill of unscored memories on startup."""
        scored, total = self.backfill_significance()
        if scored > 0:
            llm.debug_log.log(
                call_type="backfill",
                model=config.EXTRACTION_MODEL,
                prompt="Auto-backfill on startup",
                response=f"Scored {scored}/{total} memories",
                elapsed_ms=0,
            )

    # ── Declutter (time-aware, preserves timestamps) ─────────────────────

    def declutter(self) -> tuple[int, int]:
        """Deduplicate, merge, and prune memories using time + importance awareness.

        Improvements over naive declutter:
          - Includes age of each memory so the LLM can judge recency vs staleness
          - Preserves original timestamps for kept memories (doesn't reset time)
          - Asks LLM to output topic tags for consolidated facts
          - Stamps last-run date for auto-declutter scheduling
        Returns (before, after).
        """
        def _get_all_for_declutter():
            with self._lock:
                count = self._collection.count()
                if count == 0:
                    return 0, None
                return count, self._collection.get(include=["documents", "metadatas"])

        count, all_data = self._run_with_collection_retry(_get_all_for_declutter)
        if count == 0 or all_data is None:
            return 0, 0

        docs = all_data["documents"]
        metas = all_data["metadatas"]
        before = len(docs)

        # Build numbered list with age and metadata for the LLM
        lines = []
        for i, (d, m) in enumerate(zip(docs, metas)):
            sig = m.get("significance", 2)
            topic = m.get("topic", "")
            age = self._relative_time(m.get("stored_at", ""))
            access_count = m.get("access_count", 0)
            parts = [f"{i+1}. [{sig}]"]
            if topic and topic != "other":
                parts.append(f"({topic})")
            parts.append(d)
            if age:
                parts.append(f"— {age}")
            if access_count:
                parts.append(f"(recalled {access_count}x)")
            lines.append(" ".join(parts))
        numbered = "\n".join(lines)

        prompt = (
            "You are a memory manager for a personal AI companion. Below is a list of stored memories "
            "about the user, each with a significance score [1-5], optional topic, age, and recall count.\n\n"
            "Your job (think like a human brain consolidating memories during sleep):\n"
            "1. MERGE duplicates or near-duplicates into a single, richer fact preserving all details\n"
            "2. REMOVE facts that are trivial, generic, or not worth remembering long-term "
            "(e.g. 'User said hi', 'User asked about the weather')\n"
            "3. REMOVE contradicted or outdated facts if a newer one supersedes them\n"
            "4. KEEP all significant personal facts (relationships, life events, preferences, goals)\n"
            "5. KEEP frequently recalled memories (high recall count) — they matter to the user\n"
            "6. Let old trivial memories (sig 1-2, many weeks old, rarely recalled) fade away\n"
            "7. RE-SCORE each kept fact [1-5] and assign a topic tag\n\n"
            "Return the cleaned list in this exact format, one per line:\n"
            "[score|topic] fact text\n\n"
            "Topics: personal, work, health, hobby, relationship, finance, goal, opinion, other\n\n"
            'If ALL memories should be removed, return only "NONE".\n\n'
            f"Current memories:\n{numbered}"
        )

        from config import _gemini_clients, _gemini_client
        client = _gemini_clients.get(config.DECLUTTER_API_KEY, _gemini_client)
        if not client:
            raise RuntimeError("No Gemini API client available for declutter")

        model_id = config.DECLUTTER_MODEL
        t0 = time.monotonic()
        error = None
        response_text = ""
        try:
            response_text = llm.call_gemini_api(
                model_id, prompt,
                temperature=0.2, max_tokens=4096, client=client,
            )
        except Exception as e:
            error = str(e)
            raise
        finally:
            elapsed = int((time.monotonic() - t0) * 1000)
            llm.debug_log.log(
                call_type="declutter", model=model_id,
                prompt=prompt, response=response_text or "",
                elapsed_ms=elapsed, error=error,
            )

        # Parse cleaned memories
        cleaned: list[dict] = []
        if response_text.strip().upper() != "NONE":
            for line in response_text.splitlines():
                line = line.strip()
                if not line or line.upper() == "NONE":
                    continue
                # Parse [score|topic] fact
                m = re.match(r"\[(\d)\|(\w+)\]\s*(.+)", line)
                if m:
                    score = max(1, min(5, int(m.group(1))))
                    topic = m.group(2).strip().lower()
                    fact = m.group(3).strip()
                else:
                    # Fallback: [score] fact
                    m2 = re.match(r"\[(\d)\]\s*(.+)", line)
                    if m2:
                        score = max(1, min(5, int(m2.group(1))))
                        fact = m2.group(2).strip()
                    else:
                        fact = line.lstrip("•-* 0123456789.)")
                        score = 2
                    topic = "other"
                if fact and len(fact) > 2:
                    cleaned.append({"fact": fact, "significance": score, "topic": topic})

        # Build a lookup of old facts -> their metadata (to preserve timestamps)
        old_lookup: dict[str, dict] = {}
        for d, m in zip(docs, metas):
            old_lookup[d.lower().strip()] = m

        # Pre-build batch data outside the lock (no I/O, no contention)
        ts_now = datetime.now(timezone.utc).isoformat()
        new_ids: list[str] = []
        new_docs: list[str] = []
        new_metas: list[dict] = []
        for item in cleaned:
            old_meta = old_lookup.get(item["fact"].lower().strip())
            stored_at = old_meta.get("stored_at", ts_now) if old_meta else ts_now
            access_count = old_meta.get("access_count", 0) if old_meta else 0
            emotion = old_meta.get("emotion", "neutral") if old_meta else "neutral"
            new_ids.append(str(uuid.uuid4()))
            new_docs.append(item["fact"])
            new_metas.append({
                "stored_at": stored_at,
                "significance": item["significance"],
                "topic": item["topic"],
                "emotion": emotion,
                "access_count": access_count,
                "last_accessed": old_meta.get("last_accessed", "") if old_meta else "",
            })

        # Replace documents in-place to avoid invalidating external collection handles.
        def _replace_docs_in_place():
            with self._lock:
                existing_ids = self._collection.get()["ids"]
                if existing_ids:
                    self._collection.delete(ids=existing_ids)
                if new_ids:
                    self._collection.add(
                        ids=new_ids, documents=new_docs, metadatas=new_metas,
                    )

        self._run_with_collection_retry(_replace_docs_in_place)

        # Stamp declutter date
        self._stamp_declutter()
        return before, len(cleaned)

    # ── Auto-declutter in separate process ────────────────────────────────

    @staticmethod
    def _stamp_declutter():
        """Write today's date to the declutter stamp file."""
        _DECLUTTER_STAMP.parent.mkdir(parents=True, exist_ok=True)
        _DECLUTTER_STAMP.write_text(datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    @staticmethod
    def _declutter_needed() -> bool:
        """Check if declutter hasn't been run today."""
        try:
            if not _DECLUTTER_STAMP.exists():
                return True
            last = _DECLUTTER_STAMP.read_text().strip()
            return last != datetime.now(timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            return True

    def _maybe_auto_declutter(self):
        """Spawn a separate process to run declutter if it hasn't been run today."""
        if not self._declutter_needed():
            return
        def _count():
            with self._lock:
                return self._collection.count()

        count = self._run_with_collection_retry(_count)
        if count < 10:
            # Not enough memories to bother decluttering
            return
        threading.Thread(target=self._run_declutter, daemon=True).start()

    def delete_all(self):
        def _clear():
            with self._lock:
                ids = self._collection.get()["ids"]
                if ids:
                    self._collection.delete(ids=ids)

        self._run_with_collection_retry(_clear)

    def _run_declutter(self):
        """Background thread for auto-declutter."""
        try:
            before, after = self.declutter()
            llm.debug_log.log(
                call_type="auto_declutter",
                model=config.DECLUTTER_MODEL,
                prompt="Auto-declutter (background thread)",
                response=f"{before} -> {after} memories",
                elapsed_ms=0,
            )
        except Exception:
            pass

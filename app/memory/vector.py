"""向量索引：numpy 内存余弦。几千条规模足够；接口预留可换 sqlite-vec / LanceDB。"""
from __future__ import annotations

import numpy as np

from app.store import repo


class VectorIndex:
    def __init__(self):
        self._cache: dict[str, tuple[list[str], list[str], np.ndarray]] = {}

    def invalidate(self, owner_type: str) -> None:
        self._cache.pop(owner_type, None)

    def add(self, conn, owner_type: str, owner_id: str, text: str, vector: list[float]) -> None:
        repo.upsert_vector(conn, owner_type, owner_id, text, vector)
        self.invalidate(owner_type)

    def remove(self, conn, owner_type: str, owner_id: str) -> None:
        repo.delete_vector(conn, owner_type, owner_id)
        self.invalidate(owner_type)

    def _load(self, conn, owner_type: str):
        if owner_type in self._cache:
            return self._cache[owner_type]
        rows = repo.all_vectors(conn, owner_type)
        ids, texts, mats = [], [], []
        for r in rows:
            arr = np.frombuffer(r["vector"], dtype=np.float32)
            if arr.size == 0:
                continue
            ids.append(r["owner_id"])
            texts.append(r["text"])
            mats.append(arr)
        if mats:
            mat = np.vstack(mats).astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
        else:
            mat = np.zeros((0, 1), dtype=np.float32)
        self._cache[owner_type] = (ids, texts, mat)
        return self._cache[owner_type]

    def search(self, conn, owner_type: str, query_vec: list[float], top_k: int = 5,
               min_sim: float = 0.0) -> list[tuple[str, str, float]]:
        ids, texts, mat = self._load(conn, owner_type)
        if mat.shape[0] == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        n = np.linalg.norm(q)
        if n == 0:
            return []
        q = q / n
        sims = mat @ q
        order = np.argsort(-sims)[:top_k]
        out = []
        for i in order:
            s = float(sims[i])
            if s >= min_sim:
                out.append((ids[i], texts[i], s))
        return out


_index: VectorIndex | None = None


def get_index() -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex()
    return _index

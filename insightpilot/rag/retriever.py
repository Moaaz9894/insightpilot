"""Retrieval layer. One interface, two backends:
  * LocalRetriever       - TF-IDF over policy docs (zero cloud dependencies; used in tests/dev)
  * AzureSearchRetriever - hybrid (keyword + vector) search on Azure AI Search + Azure OpenAI embeddings
get_retriever() picks Azure automatically when AZURE_SEARCH_* env vars are set."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from ..config import POLICY_DIR, azure_search_configured


@dataclass
class Chunk:
    id: str
    doc: str
    heading: str
    text: str


def load_chunks() -> list[Chunk]:
    """Split each markdown policy into one chunk per '## ' section (keeps meaning intact)."""
    chunks = []
    for f in sorted(POLICY_DIR.glob("*.md")):
        parts = re.split(r"^## ", f.read_text(), flags=re.M)
        title = parts[0].lstrip("# ").strip()
        for i, p in enumerate(parts[1:]):
            heading, _, body = p.partition("\n")
            chunks.append(Chunk(f"{f.stem}-{i}", f.stem, f"{title} / {heading.strip()}", body.strip()))
    return chunks


class LocalRetriever:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = load_chunks()
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([f"{c.heading} {c.text}" for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[dict]:
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [{**asdict(self.chunks[i]), "score": round(float(scores[i]), 4)}
                for i in order if scores[i] > 0]


class AzureSearchRetriever:
    """Hybrid retrieval. NOTE: written against azure-search-documents>=11.6; verify against a live
    Azure resource (see README 'Azure deployment')."""
    INDEX = "policies"
    DIMS = 1536

    def __init__(self):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient
        from openai import AzureOpenAI
        cred = AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
        ep = os.environ["AZURE_SEARCH_ENDPOINT"]
        self.index_client = SearchIndexClient(ep, cred)
        self.client = SearchClient(ep, self.INDEX, cred)
        self.aoai = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"))
        self.embed_model = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        r = self.aoai.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in r.data]

    def ingest(self) -> int:
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration, SearchableField, SearchField, SearchFieldDataType,
            SearchIndex, SimpleField, VectorSearch, VectorSearchProfile)
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="doc", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="heading", type=SearchFieldDataType.String),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=self.DIMS,
                        vector_search_profile_name="hnsw-profile"),
        ]
        vs = VectorSearch(algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
                          profiles=[VectorSearchProfile(name="hnsw-profile",
                                                        algorithm_configuration_name="hnsw")])
        self.index_client.create_or_update_index(SearchIndex(name=self.INDEX, fields=fields, vector_search=vs))
        chunks = load_chunks()
        vecs = self._embed([f"{c.heading}. {c.text}" for c in chunks])
        self.client.upload_documents([{**asdict(c), "vector": v} for c, v in zip(chunks, vecs)])
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        from azure.search.documents.models import VectorizedQuery
        vq = VectorizedQuery(vector=self._embed([query])[0], k_nearest_neighbors=k, fields="vector")
        res = self.client.search(search_text=query, vector_queries=[vq], top=k)
        return [{"id": r["id"], "doc": r["doc"], "heading": r["heading"], "text": r["text"],
                 "score": round(float(r["@search.score"]), 4)} for r in res]


@lru_cache(maxsize=1)
def get_retriever():
    return AzureSearchRetriever() if azure_search_configured() else LocalRetriever()


if __name__ == "__main__":  # python -m insightpilot.rag.retriever  -> index docs into Azure AI Search
    print("indexed chunks:", AzureSearchRetriever().ingest())

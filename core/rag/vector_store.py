import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

class EvidenceStore:
    def __init__(self, index_path='sentinel_ai/models/evidence_index'):
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def create_mock_evidence(self):
        # Create synthetic "Knowledge Base" of policies and user histories
        documents = [
            "Policy: Transactions over $5000 from a new device must be flagged for manual review.",
            "Policy: Multiple transactions from different countries within 24 hours is a high-indicator of Account Takeover (ATO).",
            "Policy: Transactions in 'Electronics' category exceeding 3x the user's 30-day average are high risk.",
            "User USR_1003 History: Previous account takeover reported in Jan 2026. Prefers electronics and travel.",
            "User USR_1004 History: Stable banking patterns, primarily based in Seattle, WA.",
            "Fraud Pattern: 'Velocity Attack' involves 5+ transactions within 1 hour using different merchant categories."
        ]
        self.vector_store = FAISS.from_texts(documents, self.embeddings)
        self.vector_store.save_local(self.index_path)

    def load(self):
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)

    def query(self, query_text, k=2):
        if not self.vector_store:
            self.load()
        docs = self.vector_store.similarity_search(query_text, k=k)
        return [doc.page_content for doc in docs]

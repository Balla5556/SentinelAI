from typing import List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from .schemas import AgentState
from ..rag.vector_store import EvidenceStore

class FraudInvestigator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.evidence_store = EvidenceStore()
        self.evidence_store.create_mock_evidence() # In production, this would be a real DB

    def retrieve_evidence_node(self, state: AgentState):
        print("🔍 Agent: Retrieving evidence from Knowledge Base...")
        
        # Construct a query based on SHAP drivers
        drivers = ", ".join([f"{feat} ({val:.2f})" for feat, val in state['shap_evidence']])
        query = f"Fraud indicators: {drivers}. Search for relevant policies and user history."
        
        docs = self.evidence_store.query(query)
        return {"retrieved_docs": docs}

    def analyze_evidence_node(self, state: AgentState):
        print("🧠 Agent: Analyzing evidence and reasoning...")
        
        prompt = f"""
        You are a Senior Fraud Forensic Investigator.
        Transaction ID: {state['transaction_id']}
        ML Risk Score: {state['risk_score']:.2f}
        ML Drivers (SHAP): {state['shap_evidence']}
        Retrieved Evidence: {state['retrieved_docs']}
        
        Based on the ML score and the retrieved policies/history, provide a concise 
        forensic analysis. Does the evidence support the ML risk score?
        """
        response = self.llm.invoke(prompt)
        return {"investigation_notes": [response.content]}

    def triage_node(self, state: AgentState):
        print("⚖️ Agent: Determining final recommendation...")
        
        prompt = f"""
        Analyze the following investigation notes and decide the action:
        Notes: {state['investigation_notes']}
        
        Respond ONLY with a JSON-like format:
        Recommendation: [APPROVE/REJECT/ESCALATE]
        Reason: [One sentence explanation]
        """
        response = self.llm.invoke(prompt)
        content = response.content
        
        # Simple parsing for the demo
        rec = "ESCALATE"
        if "APPROVE" in content.upper(): rec = "APPROVE"
        elif "REJECT" in content.upper(): rec = "REJECT"
        
        return {"recommendation": rec, "final_report": content}

    def build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("retrieve", self.retrieve_evidence_node)
        workflow.add_node("analyze", self.analyze_evidence_node)
        workflow.add_node("triage", self.triage_node)
        
        # Define Edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", "triage")
        workflow.add_edge("triage", END)
        
        return workflow.compile()

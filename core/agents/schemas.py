from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    # Input from ML Layer
    transaction_id: str
    risk_score: float
    shap_evidence: List[tuple] # (feature_name, value)
    
    # Investigation progress
    retrieved_docs: Annotated[List[str], operator.add]
    investigation_notes: Annotated[List[str], operator.add]
    
    # Final Output
    final_report: str
    recommendation: str # "APPROVE", "REJECT", "ESCALATE"

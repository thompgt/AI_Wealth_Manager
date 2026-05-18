from langgraph.graph import StateGraph, END
from state import AgentState
from agents.macro_sentinel import macro_sentinel_node
from agents.quant_builder import quant_builder_node
from agents.tax_architect import tax_architect_node
from agents.compliance_critic import compliance_critic_node

def route_after_tax(state: AgentState):
    if state.get("needs_rebuild"):
        return "quant_builder"
    return "compliance_critic"

workflow = StateGraph(AgentState)

workflow.add_node("macro_sentinel", macro_sentinel_node)
workflow.add_node("quant_builder", quant_builder_node)
workflow.add_node("tax_architect", tax_architect_node)
workflow.add_node("compliance_critic", compliance_critic_node)

workflow.set_entry_point("macro_sentinel")
workflow.add_edge("macro_sentinel", "quant_builder")
workflow.add_edge("quant_builder", "tax_architect")
workflow.add_conditional_edges(
    "tax_architect",
    route_after_tax,
    {
        "quant_builder": "quant_builder",
        "compliance_critic": "compliance_critic"
    }
)
workflow.add_edge("compliance_critic", END)

app_workflow = workflow.compile()

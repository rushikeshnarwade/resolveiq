from langgraph.graph import StateGraph, START, END
from states import AnalyzerState
from nodes import summarize_ticket_node, retrieve_historical_context, generate_resolution_node, post_result_to_servicenow_node

workflow = StateGraph(AnalyzerState)

workflow.add_node("summarize_ticket", summarize_ticket_node)
workflow.add_node("retrieve_context", retrieve_historical_context)
workflow.add_node("generate_solution", generate_resolution_node)
workflow.add_node("post_to_servicenow", post_result_to_servicenow_node)

workflow.add_edge(START, "summarize_ticket")
workflow.add_edge("summarize_ticket", "retrieve_context")
workflow.add_edge("retrieve_context", "generate_solution")
workflow.add_edge("generate_solution", "post_to_servicenow")
workflow.add_edge("post_to_servicenow", END)

workflow = workflow.compile()
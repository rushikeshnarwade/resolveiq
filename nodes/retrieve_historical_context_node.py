from states import AnalyzerState
from utils import get_embeddings, get_vector_store


def retrieve_historical_context(state: AnalyzerState) -> dict:
    ticket = state["ticket"]
    search_query = (state["clean_summary"] or ticket.short_description).strip()

    filter_dict = {}
    if ticket.cmdb_ci:
        filter_dict["cmdb_ci"] = ticket.cmdb_ci

    embeddings = get_embeddings()
    vector_store = get_vector_store()
    query_embedding = embeddings.embed_query(search_query)

    result = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=3,
        filter=filter_dict if filter_dict else None,
    )

    past_solutions = [doc.page_content for doc in result]
    return {"past_solutions": past_solutions}
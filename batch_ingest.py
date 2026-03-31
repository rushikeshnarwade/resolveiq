import os
import asyncio
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres.vectorstores import PGVector
from dotenv import load_dotenv
import logging

load_dotenv()

# ==========================================
# 1. SETUP CONNECTIONS
# ==========================================
DB_CONNECTION = os.getenv("PGVECTOR_URL")
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="historical_tickets",
    connection=DB_CONNECTION,
    use_jsonb=True,
)

summarizer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 💡 UPGRADE: Enriched prompt with your suggested metadata
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an IT knowledge engineer. Extract the core technical issue and the final resolution into a dense, brief technical summary. Use the provided metadata (CI, Category) to add context if helpful."),
    ("human", "Short Description: {short_desc}\nFull Description: {desc}\nCategory: {category}\nConfiguration Item: {cmdb_ci}\n\nResolution Notes: {resolution}")
])
summarization_chain = summary_prompt | summarizer_llm | StrOutputParser()

# ==========================================
# 2. THE BATCH INGESTION FUNCTION (No Mock Data)
# ==========================================
async def process_batch(resolved_tickets: list):
    """
    Takes a list of validated Pydantic ticket models, summarizes them, 
    and upserts them into the pgvector database.
    """
    logging.info(f"🚀 Starting batch ingestion for {len(resolved_tickets)} tickets...")
    
    documents_to_insert = []
    document_ids = []

    for ticket in resolved_tickets:
        logging.info(f"⏳ Processing {ticket.number}...")
        
        # Injecting all the context you requested
        clean_summary = summarization_chain.invoke({
            "short_desc": ticket.short_description,
            "desc": ticket.description,
            "category": ticket.category,
            "cmdb_ci": ticket.cmdb_ci or "Unknown",
            "resolution": ticket.close_notes
        })
        
        doc = Document(
            page_content=clean_summary,
            metadata={
                "sys_id": ticket.sys_id,
                "number": ticket.number,
                "cmdb_ci": ticket.cmdb_ci,
                "category": ticket.category
            }
        )
        documents_to_insert.append(doc)
        document_ids.append(ticket.sys_id) # We track the sys_id for the upsert
        
        logging.info(f"✅ {ticket.number} summarized.")

    # 💡 UPGRADE: By passing 'ids', PGVector will overwrite duplicates instead of crashing!
    logging.info("💾 Saving embeddings to database...")
    vector_store.add_documents(documents=documents_to_insert, ids=document_ids)
    logging.info("🎉 Batch ingestion complete!")
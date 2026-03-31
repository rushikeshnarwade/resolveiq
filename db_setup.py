import os
import psycopg2
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)

def setup_database():
    """Safely initializes the pgvector database schema."""
    
    # 1. Pull connection details from .env (NEVER hardcode in production!)
    # Fallback to local docker credentials only if the .env variable is missing
    db_url = os.getenv(
        "DATABASE_URL", 
        "postgresql://admin:admin@localhost:5432/analyzer"
    )
    
    try:
        logging.info("Connecting to database...")
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cur = conn.cursor()

        # 2. Enable the vector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logging.info("✅ pgvector extension verified.")

        # 3. Safely create the table ONLY if it doesn't exist.
        # Notice we removed the DROP TABLE command!
        cur.execute("""
            CREATE TABLE IF NOT EXISTS historical_tickets (
                sys_id VARCHAR PRIMARY KEY,
                clean_summary TEXT,
                embedding vector(768),
                cmetadata JSONB
            );
        """)
        logging.info("✅ Database schema verified.")

        cur.close()
        conn.close()
        logging.info("🎉 Database setup complete!")

    except Exception as e:
        logging.error(f"❌ Database setup failed: {e}")

if __name__ == "__main__":
    setup_database()
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os

# Load environment variables
load_dotenv()

def load_data_from_snowflake(query: str) -> pd.DataFrame:
    """
    Executes a SQL query on Snowflake and returns a Pandas DataFrame.
    """
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    role = os.getenv("SNOWFLAKE_ROLE")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")

    # ✅ SQLAlchemy Snowflake connection string (using snowflake-sqlalchemy)
    conn_str = (
        f"snowflake://{user}:{password}@{account}/"
        f"{database}/{schema}?warehouse={warehouse}&role={role}"
    )

    # Create engine
    engine = create_engine(conn_str)

    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        raise RuntimeError(f"❌ Snowflake query failed: {e}")

    return df

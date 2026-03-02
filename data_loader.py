from sqlalchemy import create_engine, text
import pandas as pd


def fetch_from_sql(connection_string, table_name="azure_devops_bugs", schema="dbo"):
    """Fetches work items from SQL Server."""
    try:
        engine = create_engine(connection_string)
        query = f"SELECT * FROM {schema}.{table_name}"
        
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
            
        return df.to_dict('records')
    except Exception as e:
        print(f"Error fetching from SQL: {e}")
        return []

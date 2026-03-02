from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from config import DB_CONNECTION_STRING, TABLE_NAME

def get_db():
    engine = create_engine(DB_CONNECTION_STRING)
    return SQLDatabase(engine,include_tables=TABLE_NAME)

import os

# Database Configuration
TABLE_NAME = "azure_devops_bugs"
SCHEMA_NAME = "dbo"
DB_CONNECTION_STRING = (
    "mssql+pyodbc://localhost/Project?"
    "driver=ODBC+Driver+17+for+SQL+Server&"
    "trusted_connection=yes&"
    "TrustServerCertificate=yes"
)

# Application Settings
MEMORY_FILE = "query_memory.json"
OLLAMA_MODEL = "qwen2.5-coder:7b"

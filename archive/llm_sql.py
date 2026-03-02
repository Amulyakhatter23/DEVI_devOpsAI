import os
import json
import pandas as pd
# import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
# from langchain.chains import create_sql_query_chain # Removed due to import error
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Configuration
GROQ_API_KEY = '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'
TABLE_NAME = "Azure devOps"
SCHEMA_NAME = "dbo" # SQLAlchemy usually handles schema without brackets in connection string, but we might need it for table inclusion
MEMORY_FILE = "query_memory.json"

# Set API Key for LangChain
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def get_db_engine():
    """Creates a SQLAlchemy engine for the SQL Server connection."""
    # Connection string for MSSQL with Windows Authentication (Trusted_Connection=yes)
    # Using mssql+pyodbc driver
    # Note: You might need to install pyodbc: pip install pyodbc
    connection_string = (
        "mssql+pyodbc://localhost/Project?"
        "driver=ODBC+Driver+17+for+SQL+Server&"
        "trusted_connection=yes&"
        "TrustServerCertificate=yes"
    )
    engine = create_engine(connection_string)
    return engine

def load_memory():
    """Loads query memory from JSON file."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_memory(question, sql_query):
    """Saves a new question-query pair to memory."""
    memory = load_memory()
    memory[question] = sql_query
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)
    print("✅ Query saved to memory!")

def main():
    print("🤖 LangChain SQL Query Assistant (CLI)")
    print(f"Target Table: {TABLE_NAME}")
    print("Type 'exit' or 'quit' to stop.\n")

    # 1. Setup Database
    try:
        engine = get_db_engine()
        # We only include the specific table we are interested in to reduce context size
        db = SQLDatabase(engine, include_tables=[TABLE_NAME], schema=SCHEMA_NAME)
        print("✅ Connected to database.")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return

    # 2. Setup LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # 3. Setup Chains
    # Define custom prompt to enforce T-SQL rules
    template = """You are an expert T-SQL data analyst. 
    Your task is to generate a **single, syntactically correct SQL Server (T-SQL) query** to analyze **Azure DevOps Bug work items** based on the user question.

    ### Database Context
    - All records represent **Bug** work items from Azure DevOps.
    - Use **only the columns listed below**.
    - Do NOT hallucinate columns or tables.
    - Do NOT include explanations, comments, or markdown.
    - Return **ONLY the SQL query**.

    ### Table Information
    {table_info}

    ### Column Definitions
    1. Id - Unique identifier for each Azure DevOps bug
    2. Title - Short summary of the bug
    3. WorkItemType - Always 'Bug'
    4. State - Current workflow state of the bug
    5. AssignedTo - Person responsible for the bug
    6. Description - Detailed bug description
    7. Created_Date - Date when the bug was created
    8. Completed_Date - Date when the bug was closed;  
        if NULL, treat it as **GETDATE()**
    9. Severity – Integer value defining business impact (1 = highest severity, 4 = lowest severity).
    10. Priority - Urgency of fixing the bug (1 = High/Critical, 2 = Medium, 3 = Low)
    11. Module - Functional area where the bug was raised
    12. Environment - One of:
        (Performance, Security, Production, Regression, PreProd, Integration Testing, POD)

    ### Query Rules
    - Use **T-SQL syntax only**
    - Use `ISNULL(Changed_date, GETDATE())` when calculating durations
    - Use proper `GROUP BY` when aggregations are involved
    - Use `TOP`, `ORDER BY`, and `WHERE` clauses only when relevant
    - Assume dates are stored in SQL Server `DATETIME` format
    - Use `ISNULL(Changed_Date, GETDATE())` when calculating durations
    - Use proper `GROUP BY` when aggregations are involved
    - Use `TOP`, `ORDER BY`, and `WHERE` clauses only when relevant
    - Assume dates are stored in SQL Server `DATETIME` format
    - Don't Apply any filter condition unless specified
    - **CRITICAL**: When asking for "Last Month", do NOT use `ELSE 'Last Month'`. You MUST explicitly filter for the previous month (e.g., `DATEDIFF(month, [DateCol], GETDATE()) = 1`).
    - **CRITICAL**: When comparing "This Month" vs "Last Month", ensure "Last Month" ONLY includes data from the previous month, not all history.
    - **CRITICAL**: You MUST include a `WHERE` clause that filters for the specific time periods requested. Do NOT rely on `CASE WHEN` alone to separate data. For example, `WHERE DATEDIFF(month, [DateCol], GETDATE()) IN (0, 1)`.

    Question: {input}
    SQL Query:"""
    prompt = PromptTemplate.from_template(template)

    # Chain to generate SQL using LCEL
    def get_schema(_):
        return db.get_table_info()

    generate_query = (
        RunnablePassthrough.assign(table_info=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Chain to generate Plotting Code
    # plot_template = """You are a data visualization expert.
    # The user asked: {question}
    # 
    # I have a pandas DataFrame named `df` with the following columns and data types:
    # {dtypes}
    # 
    # Here are the first 5 rows:
    # {head}
    # 
    # Write Python code using `matplotlib.pyplot` (as plt) to visualize this data.
    # Rules:
    # 1. Use `df` directly. DO NOT create sample data.
    # 2. Set a title and labels.
    # 3. Use `plt.show()` at the end.
    # Rules:
    # 1. Use `df` directly. DO NOT create sample data.
    # 2. Set a title and labels.
    # 3. Use `plt.show()` at the end.
    # 4. If you need to extract date parts (year, month, etc.), convert the column to datetime first using `pd.to_datetime(df['col'])`.
    # 5. Handle timezones: use `.dt.tz_localize(None)` on datetime columns to avoid tz-aware vs tz-naive errors.
    # 6. Return ONLY the Python code. No markdown, no explanations.
    # 
    # Python Code:"""
    # plot_prompt = PromptTemplate.from_template(plot_template)
    # 
    # generate_plot = (
    #     plot_prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # Chain to Review and Correct SQL
    review_template = """You are a SQL Reviewer.
    Your job is to check the following SQL query for correctness and validity given the user's question and the table schema.
    
    Table Schema:
    {table_info}
    
    User Question: {input}
    Generated SQL: {sql_query}
    
    Rules:
    1. Check if the SQL answers the question correctly.
    2. Check for T-SQL syntax errors.
    3. Check if the columns exist in the schema.
    4. Check for logical errors in date comparisons (e.g., ensuring 'Last Month' is actually last month and not just 'everything else').
    5. If the SQL is correct, output the EXACT same SQL.
    6. If the SQL is incorrect, output the CORRECTED SQL.
    7. Return ONLY the SQL query. No markdown, no explanations.
    
    Reviewed SQL:"""
    review_prompt = PromptTemplate.from_template(review_template)
    
    review_chain = (
        RunnablePassthrough.assign(table_info=get_schema)
        | review_prompt
        | llm
        | StrOutputParser()
    )

    # Tool to execute SQL
    execute_query = QuerySQLDataBaseTool(db=db)

    while True:
        try:
            question = input("\n📝 Enter your question: ").strip()
        except EOFError:
            break

        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if not question:
            continue

        # Check Memory First
SCHEMA_NAME = "dbo" # SQLAlchemy usually handles schema without brackets in connection string, but we might need it for table inclusion
MEMORY_FILE = "query_memory.json"

# Set API Key for LangChain
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def get_db_engine():
    """Creates a SQLAlchemy engine for the SQL Server connection."""
    # Connection string for MSSQL with Windows Authentication (Trusted_Connection=yes)
    # Using mssql+pyodbc driver
    # Note: You might need to install pyodbc: pip install pyodbc
    connection_string = (
        "mssql+pyodbc://localhost/Project?"
        "driver=ODBC+Driver+17+for+SQL+Server&"
        "trusted_connection=yes&"
        "TrustServerCertificate=yes"
    )
    engine = create_engine(connection_string)
    return engine

def load_memory():
    """Loads query memory from JSON file."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_memory(question, sql_query):
    """Saves a new question-query pair to memory."""
    memory = load_memory()
    memory[question] = sql_query
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)
    print("✅ Query saved to memory!")

def main():
    print("🤖 LangChain SQL Query Assistant (CLI)")
    print(f"Target Table: {TABLE_NAME}")
    print("Type 'exit' or 'quit' to stop.\n")

    # 1. Setup Database
    try:
        engine = get_db_engine()
        # We only include the specific table we are interested in to reduce context size
        db = SQLDatabase(engine, include_tables=[TABLE_NAME], schema=SCHEMA_NAME)
        print("✅ Connected to database.")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return

    # 2. Setup LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # 3. Setup Chains
    # Define custom prompt to enforce T-SQL rules
    template = """You are an expert T-SQL data analyst. 
    Your task is to generate a **single, syntactically correct SQL Server (T-SQL) query** to analyze **Azure DevOps Bug work items** based on the user question.

    ### Database Context
    - All records represent **Bug** work items from Azure DevOps.
    - Use **only the columns listed below**.
    - Do NOT hallucinate columns or tables.
    - Do NOT include explanations, comments, or markdown.
    - Return **ONLY the SQL query**.

    ### Table Information
    {table_info}

    ### Column Definitions
    1. Id - Unique identifier for each Azure DevOps bug
    2. Title - Short summary of the bug
    3. WorkItemType - Always 'Bug'
    4. State - Current workflow state of the bug
    5. AssignedTo - Person responsible for the bug
    6. Description - Detailed bug description
    7. Created_Date - Date when the bug was created
    8. Completed_Date - Date when the bug was closed;  
        if NULL, treat it as **GETDATE()**
    9. Severity – Integer value defining business impact (1 = highest severity, 4 = lowest severity).
    10. Priority - Urgency of fixing the bug
    11. Module - Functional area where the bug was raised
    12. Environment - One of:
        (Performance, Security, Production, Regression, PreProd, Integration Testing, POD)

    ### Query Rules
    - Use **T-SQL syntax only**
    - Use `ISNULL(Changed_date, GETDATE())` when calculating durations
    - Use proper `GROUP BY` when aggregations are involved
    - Use `TOP`, `ORDER BY`, and `WHERE` clauses only when relevant
    - Assume dates are stored in SQL Server `DATETIME` format
    - Use `ISNULL(Changed_Date, GETDATE())` when calculating durations
    - Use proper `GROUP BY` when aggregations are involved
    - Use `TOP`, `ORDER BY`, and `WHERE` clauses only when relevant
    - Assume dates are stored in SQL Server `DATETIME` format
    - Don't Apply any filter condition unless specified
    - **CRITICAL**: When asking for "Last Month", do NOT use `ELSE 'Last Month'`. You MUST explicitly filter for the previous month (e.g., `DATEDIFF(month, [DateCol], GETDATE()) = 1`).
    - **CRITICAL**: When comparing "This Month" vs "Last Month", ensure "Last Month" ONLY includes data from the previous month, not all history.
    - **CRITICAL**: You MUST include a `WHERE` clause that filters for the specific time periods requested. Do NOT rely on `CASE WHEN` alone to separate data. For example, `WHERE DATEDIFF(month, [DateCol], GETDATE()) IN (0, 1)`.

    Question: {input}
    SQL Query:"""
    prompt = PromptTemplate.from_template(template)

    # Chain to generate SQL using LCEL
    def get_schema(_):
        return db.get_table_info()

    generate_query = (
        RunnablePassthrough.assign(table_info=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Chain to generate Plotting Code
    # plot_template = """You are a data visualization expert.
    # The user asked: {question}
    # 
    # I have a pandas DataFrame named `df` with the following columns and data types:
    # {dtypes}
    # 
    # Here are the first 5 rows:
    # {head}
    # 
    # Write Python code using `matplotlib.pyplot` (as plt) to visualize this data.
    # Rules:
    # 1. Use `df` directly. DO NOT create sample data.
    # 2. Set a title and labels.
    # 3. Use `plt.show()` at the end.
    # Rules:
    # 1. Use `df` directly. DO NOT create sample data.
    # 2. Set a title and labels.
    # 3. Use `plt.show()` at the end.
    # 4. If you need to extract date parts (year, month, etc.), convert the column to datetime first using `pd.to_datetime(df['col'])`.
    # 5. Handle timezones: use `.dt.tz_localize(None)` on datetime columns to avoid tz-aware vs tz-naive errors.
    # 6. Return ONLY the Python code. No markdown, no explanations.
    # 
    # Python Code:"""
    # plot_prompt = PromptTemplate.from_template(plot_template)
    # 
    # generate_plot = (
    #     plot_prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # Chain to Review and Correct SQL
    review_template = """You are a SQL Reviewer.
    Your job is to check the following SQL query for correctness and validity given the user's question and the table schema.
    
    Table Schema:
    {table_info}
    
    User Question: {input}
    Generated SQL: {sql_query}
    
    Rules:
    1. Check if the SQL answers the question correctly.
    2. Check for T-SQL syntax errors.
    3. Check if the columns exist in the schema.
    4. Check for logical errors in date comparisons (e.g., ensuring 'Last Month' is actually last month and not just 'everything else').
    5. If the SQL is correct, output the EXACT same SQL.
    6. If the SQL is incorrect, output the CORRECTED SQL.
    7. Return ONLY the SQL query. No markdown, no explanations.
    
    Reviewed SQL:"""
    review_prompt = PromptTemplate.from_template(review_template)
    
    review_chain = (
        RunnablePassthrough.assign(table_info=get_schema)
        | review_prompt
        | llm
        | StrOutputParser()
    )

    # Tool to execute SQL
    execute_query = QuerySQLDataBaseTool(db=db)

    while True:
        try:
            question = input("\n📝 Enter your question: ").strip()
        except EOFError:
            break

        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if not question:
            continue

        # Check Memory First
        memory = load_memory()
        if question in memory:
            print("💡 Query retrieved from memory!")
            sql_query = memory[question]
            from_memory = True
            # We still need to enter the loop to execute, but we'll skip generation unless feedback forces a retry
            # Actually, if it's from memory, we just execute. If feedback is bad, we might want to drop from memory and retry?
            # For simplicity, let's treat memory hit as a single attempt. If wrong, we start fresh generation.
        else:
            from_memory = False

        user_feedback = None
        
        while True: # Retry Loop
            if from_memory and not user_feedback:
                # Just use the memory query
                pass 
            else:
                print("⏳ Generating SQL query")
                try:
                    # Prepare input with feedback if exists
                    chain_input = question
                    if user_feedback:
                        chain_input += f"\n\nIMPORTANT FEEDBACK FROM PREVIOUS ATTEMPT: {user_feedback}\nFix the query based on this feedback."
                    
                    # Generate SQL
                    # LCEL chain expects a dict with 'input' key based on our prompt
                    response = generate_query.invoke({"input": chain_input})
                    # Clean up the response if it contains markdown code blocks (LangChain sometimes leaves them)
                    sql_query = response.strip()
                    if sql_query.startswith("```sql"):
                        sql_query = sql_query[6:]
                    if sql_query.endswith("```"):
                        sql_query = sql_query[:-3]
                    sql_query = sql_query.strip()
                    
                    # Review SQL
                    print("🔍 Reviewing SQL...")
                    sql_query = review_chain.invoke({
                        "input": chain_input,
                        "sql_query": sql_query
                    })
                    # Clean up reviewed SQL
                    sql_query = sql_query.strip()
                    if sql_query.startswith("```sql"):
                        sql_query = sql_query[6:]
                    if sql_query.endswith("```"):
                        sql_query = sql_query[:-3]
                    sql_query = sql_query.strip()
                    
                    from_memory = False # We generated a new one
                except Exception as e:
                    print(f"❌ Error generating SQL: {e}")
                    # If generation fails, we break the retry loop to avoid infinite error loops
                    break

            print(f"\n```sql\n{sql_query}\n```\n")
    
            print("⏳ Fetching results...")
            try:
                # Execute SQL
                df = pd.read_sql(sql_query, engine)
                
                if not df.empty:
                    print(df.to_string(index=False))
                    print(f"\n(Found {len(df)} rows)")
    
                    # Feedback Loop
                    if not from_memory or user_feedback:
                        feedback = input("\nWas this correct? (y/n): ").strip().lower()
                        if feedback == 'y':
                            save_memory(question, sql_query)
                            break # Exit retry loop
                        else:
                            user_feedback = input("Please describe what was wrong: ").strip()
                            print("🔄 Retrying with feedback...")
                            from_memory = False # Ensure we generate next time
                            continue # Restart retry loop
                    else:
                        break # From memory and we assume correct (or user can't correct it easily here)
                else:
                    print("⚠️ Query returned no results.")
                    # Allow retry even if no results
                    feedback = input("\nWas this correct? (y/n): ").strip().lower()
                    if feedback == 'n':
                         user_feedback = input("Please describe what was wrong: ").strip()
                         print("🔄 Retrying with feedback...")
                         from_memory = False
                         continue
                    break
    
            except Exception as e:
                print(f"❌ Failed to execute query: {e}")
                # Allow retry on execution error
                feedback = input("\nDo you want to retry with feedback? (y/n): ").strip().lower()
                if feedback == 'y':
                     user_feedback = input("Please describe what was wrong: ").strip()
                     print("🔄 Retrying with feedback...")
                     from_memory = False
                     continue
                break

if __name__ == "__main__":
    main()

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
import difflib
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Configuration
GROQ_API_KEY = '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'
TABLE_NAME = "azure_devops_bugs"
SCHEMA_NAME = "dbo" # SQLAlchemy usually handles schema without brackets in connection string, but we might need it for table inclusion
MEMORY_FILE = "query_memory.json"

# Set API Key for LangChain
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def get_db_engine():
    """Creates a SQLAlchemy engine for the SQL
     Server connection."""
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

def get_similar_examples(question, k=3):
    """Finds the top k similar questions from memory."""
    memory = load_memory()
    if not memory:
        return ""
    
    questions = list(memory.keys())
    matches = difflib.get_close_matches(question, questions, n=k, cutoff=0.4)
    
    if not matches:
        return ""
    
    examples = ""
    for q in matches:
        examples += f"Q: {q}\nSQL: {memory[q]}\n\n"
    
    return examples

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
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

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

    ### Similar Past Examples
    Here are some examples of correct SQL queries for similar questions
    (Note:- Use these examples ONLY as structural guidance,
    Do NOT reuse filters or date conditions unless explicitly requested.):
    {examples}

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
    9. Severity - Integer value defining business impact (1 = highest severity, 4 = lowest severity).
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
    - **CRITICAL**: When asking for "Last Month", do NOT use `ELSE 'Last Month'`. You MUST explicitly filter for the previous month (e.g., `DATEDIFF(month, [DateCol], GETDATE()) = 1`).
    - **CRITICAL**: When comparing "This Month" vs "Last Month", ensure "Last Month" ONLY includes data from the previous month, not all history.
    - **CRITICAL**: You MUST include a `WHERE` clause that filters for the specific time periods requested. Do NOT rely on `CASE WHEN` alone to separate data. For example, `WHERE DATEDIFF(month, [DateCol], GETDATE()) IN (0, 1)`.
    - **CRITICAL**: Dont use 'WHERE' condition unless explicit user prompting

    Question: {input}
    SQL Query:"""
    prompt = PromptTemplate.from_template(template)

    # Chain to generate SQL using LCEL
    def get_schema(_):
        return db.get_table_info()

    def get_examples(inputs):
        return get_similar_examples(inputs["input"])

    generate_query = (
        RunnablePassthrough.assign(
            table_info=get_schema,
            examples=get_examples
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chain to generate SQL using LCEL
    def get_schema(_):
        return db.get_table_info()

    def get_examples(inputs):
        return get_similar_examples(inputs["input"])

    generate_query = (
        RunnablePassthrough.assign(
            table_info=get_schema,
            examples=get_examples
        )
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

    
    7. Return ONLY the SQL query. No markdown, no explanations.
    8. Do NOT include "Corrected SQL Query:" or any other text. JUST THE SQL.
    
    Reviewed SQL:"""
    review_prompt = PromptTemplate.from_template(review_template)
    
    review_chain = (
        RunnablePassthrough.assign(table_info=get_schema)
        | review_prompt
        | llm
        | StrOutputParser()
    )

    def extract_sql(text):
        """Extracts SQL from markdown code blocks or returns raw text."""
        # Look for ```sql ... ``` blocks
        matches = re.findall(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if matches:
            # Return the last match as it's likely the corrected one if multiple exist
            return matches[-1].strip()
        
        # Look for generic ``` ... ``` blocks
        matches = re.findall(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if matches:
            return matches[-1].strip()
            
        return text.strip()

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
        else:
            print("⏳ Generating SQL query")

            try:
                # Generate SQL
                # LCEL chain expects a dict with 'input' key based on our prompt
                # Examples are now fetched within the chain
                
                # Optional: Print if examples found (we'd need to check outside or add a callback, 
                # but for now let's trust the chain)
                examples = get_similar_examples(question)
                if examples:
                    print(f"📚 Found similar examples from memory.")

                response = generate_query.invoke({"input": question})
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
                    "input": question,
                    "sql_query": sql_query
                })
                # Clean up reviewed SQL
                sql_query = extract_sql(sql_query)
                
                from_memory = False
            except Exception as e:
                print(f"❌ Error generating SQL: {e}")
                continue

        print(f"\n```sql\n{sql_query}\n```\n")

        print("⏳ Fetching results...")
        try:
            # Execute SQL
            # We can use the execute_query tool directly, or db.run()
            # execute_query returns a string representation of the result usually
            # For a DataFrame, we might want to use pandas directly with the engine
            
            # Let's use pandas for better formatting as in the original script
            df = pd.read_sql(sql_query, engine)
            
            if not df.empty:
                print(df.to_string(index=False))
                print(f"\n(Found {len(df)} rows)")

                # Try to plot if reasonable size
                # if len(df) > 0:
                #     try:
                #         print("📊 Generating chart...")
                #         plot_code = generate_plot.invoke({
                #             "question": question,
                #             "dtypes": str(df.dtypes),
                #             "head": str(df.head().to_markdown())
                #         })
                #         
                #         # Clean code
                #         plot_code = plot_code.strip()
                #         if plot_code.startswith("```python"):
                #             plot_code = plot_code[9:]
                #         if plot_code.endswith("```"):
                #             plot_code = plot_code[:-3]
                #         plot_code = plot_code.strip()
                #         
                #         # Execute plot code
                #         # We pass 'df' and 'plt' to the exec environment
                #         # exec(plot_code, {'df': df, 'plt': plt})
                #         print("✅ Chart displayed.")
                #     except Exception as e:
                #         print(f"⚠️ Could not generate chart: {e}")

                # Feedback Loop (only if not from memory)
                if not from_memory:
                    feedback = input("\nWas this correct? (y/n): ").strip().lower()
                    if feedback == 'y':
                        save_memory(question, sql_query)
            else:
                print("⚠️ Query returned no results.")

        except Exception as e:
            print(f"❌ Failed to execute query: {e}")

if __name__ == "__main__":
    main()

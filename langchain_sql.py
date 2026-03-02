import os
import json
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import difflib
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
from ollama_service import ollama_service

from chart_agent import ChartAgent
import config

"""Creates a SQLAlchemy engine for the SQL Server connection."""
def get_db_engine():
    engine = create_engine(config.DB_CONNECTION_STRING)
    return engine

"""Loads query memory from query_memory.json file."""
def load_memory():
    if os.path.exists(config.MEMORY_FILE):
        try:
            with open(config.MEMORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

"""Saves a new question-query pair to the query_memory.json file"""
def save_memory(question, sql_query):
    memory = load_memory()
    memory[question] = sql_query
    with open(config.MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)
    print("Query saved to memory!")

"""Finds the top k similar questions from memory."""
def get_similar_examples(question, k=4):
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

def get_app_components():
    """Initializes and returns the application components."""
    # 1. Setup Database
    try:
        engine = get_db_engine()
        # We only include the specific table we are interested in to reduce context size
        db = SQLDatabase(engine, include_tables=[config.TABLE_NAME], schema=config.SCHEMA_NAME)
        print("Connected to database.")
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return None

    # 2. Setup LLM
    llm = ollama_service.get_chat_model(temperature=0)

    # 3. Setup Chains
    # Define custom prompt to enforce T-SQL rules
    template = """You are an expert T-SQL data analyst.

Your task is to generate a single, syntactically correct SQL Server (T-SQL) query to analyze Azure DevOps Bug work items based on the user’s question.

Return ONLY the SQL query.
Do NOT include explanations, comments, markdown, or extra text.

---------------------------------------------------------------------

### DATABASE CONTEXT

- All records represent Bug work items from Azure DevOps.
- Use ONLY the columns listed below.
- Do NOT hallucinate tables or columns.
- Generate exactly one valid T-SQL query.

### TABLE SCHEMA
{table_info}

### SIMILAR PAST EXAMPLES
The following examples are structurally correct.
Use them only for structural guidance.
Do NOT reuse filters, conditions, or date logic unless explicitly requested by the user.

{examples}

---------------------------------------------------------------------

### COLUMN DEFINITIONS

1. Id- Unique identifier for each bug
2. Title- Short summary of the bug
3. WorkItemType- Always 'Bug'
4. State- Current workflow state
5. AssignedTo- Person responsible
6. Description- Detailed bug description
7. Created_Date- Bug creation date (DATETIME)
8. Completed_Date- Bug closure date (DATETIME)
   - If NULL, treat as GETDATE() when calculating durations
9. Severity- Integer (1 = highest severity, 4 = lowest severity)
10. Priority- Urgency level
11. Module- Functional area
12. Environment- One of:
   (Performance, Security, Production, Regression, PreProd, Integration Testing, POD)

---------------------------------------------------------------------

### QUERY RULES

1. Use T-SQL syntax only (SQL Server compatible).
2. If calculating resolution duration, use:
   DATEDIFF(day, Created_Date, ISNULL(Completed_Date, GETDATE()))

3. Use GROUP BY whenever aggregate functions are used.
4. Use TOP only when ranking or limiting results is explicitly required.
   - TOP must appear immediately after SELECT.
5. Use ORDER BY only when logically required.

6. Do NOT apply filters unless explicitly requested in the user question.

7. Time-Based Filtering Rules:
   - When the question specifies time periods (e.g., "This Month", "Last Month"),
     you MUST include a WHERE clause filtering for that period.
   - For "Last Month", explicitly filter:
       DATEDIFF(month, <DateColumn>, GETDATE()) = 1
   - For comparisons like "This Month vs Last Month",
       use:
       WHERE DATEDIFF(month, <DateColumn>, GETDATE()) IN (0, 1)
   - Do NOT rely solely on CASE statements for time separation.

8. Severity Logic:
   - Highest severity = lowest numeric value (Severity ASC).
   - Lowest severity = highest numeric value (Severity DESC).

---------------------------------------------------------------------

### USER QUESTION
{input}

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

    # Chain to Review and Correct SQL
    review_template = """
You are a SQL Reviewer responsible for validating and correcting a generated T-SQL query.

Your task:
- Check whether the SQL query correctly answers the user's question.
- Ensure it strictly follows the provided schema and rules.
- If the query is incorrect, rewrite it correctly.
- If correct, return it as-is.

Table Schema:
{table_info}

User Question:
{input}

Generated SQL:
{sql_query}

-----------------------------------------------------
COLUMN DEFINITIONS
-----------------------------------------------------
1. Id - Unique identifier for each Azure DevOps bug
2. Title - Short summary of the bug
3. WorkItemType - Always 'Bug'
4. State - Current workflow state of the bug
5. AssignedTo - Person responsible for the bug
6. Description - Detailed bug description
7. Created_Date - Date when the bug was created
8. Completed_Date - Date when the bug was closed; if NULL treat as GETDATE()
9. Severity - Integer (1 = highest severity, 4 = lowest severity)
10. Priority - Urgency of fixing the bug
11. Module - Functional area where the bug was raised
12. Environment - One of:
   (Performance, Security, Production, Regression, PreProd, Integration Testing, POD)

-----------------------------------------------------
STRICT QUERY RULES
-----------------------------------------------------
1. Use T-SQL syntax only.
2. TOP must appear immediately after SELECT (e.g., SELECT TOP 5 ...).
3. Use proper GROUP BY when aggregation functions are present.
4. Do NOT apply any filter conditions unless explicitly requested.
5. Dates are stored in SQL Server DATETIME format.
6. When calculating durations, use:
   ISNULL(Completed_Date, GETDATE())
7. For "Highest Severity", sort by Severity ASC.
8. For "Lowest Severity", sort by Severity DESC.

-----------------------------------------------------
CRITICAL TIME FILTER RULES
-----------------------------------------------------
1. When asked for "Last Month", explicitly filter:
   DATEDIFF(month, <DateColumn>, GETDATE()) = 1
2. When comparing "This Month" vs "Last Month",
   you MUST use a WHERE clause such as:
   DATEDIFF(month, <DateColumn>, GETDATE()) IN (0, 1)
3. Do NOT use CASE WHEN alone to separate time periods.
4. Always include a proper WHERE clause for time-based filtering.

-----------------------------------------------------
OUTPUT REQUIREMENTS
-----------------------------------------------------
- Return ONLY the corrected SQL query.
- Do NOT include explanations.
- Do NOT include markdown.
- Do NOT prefix with labels like "Corrected SQL".
- Output must contain only valid T-SQL.
"""
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

    return {
        "engine": engine,
        "db": db,
        "generate_query": generate_query,
        "review_chain": review_chain,
        "extract_sql": extract_sql
    }

def main():
    print("LangChain SQL Query Assistant (CLI)")
    print(f"Target Table: {config.TABLE_NAME}")
    print("Type 'exit' or 'quit' to stop.\n")

    components = get_app_components()
    if not components:
        return

    engine = components["engine"]
    generate_query = components["generate_query"]
    review_chain = components["review_chain"]
    extract_sql = components["extract_sql"]

    while True:
        try:
            question = input("\n[INPUT] Enter your question: ").strip()
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
            print("Query retrieved from memory!")
            sql_query = memory[question]
            from_memory = True
        else:
            print("Generating SQL query")

            try:
                # Generate SQL
                examples = get_similar_examples(question)
                if examples:
                    print(f"Found similar examples from memory.")

                response = generate_query.invoke({"input": question})
                sql_query = extract_sql(response)
                
                # Review SQL
                print("Reviewing SQL...")
                sql_query = review_chain.invoke({
                    "input": question,
                    "sql_query": sql_query
                })
                sql_query = extract_sql(sql_query)
                
                from_memory = False
            except Exception as e:
                print(f"Error generating SQL: {e}")
                continue

        print(f"\n```sql\n{sql_query}\n```\n")

        print("Fetching results...")
        try:
            df = pd.read_sql(sql_query, engine)
            
            if not df.empty:
                print(df.to_string(index=False))
                print(f"\n(Found {len(df)} rows)")

                # Try to plot if reasonable size
                if len(df) > 0:
                    try:
                        vis_input = input("\nDo you want to visualize this? (y/n): ").strip().lower()
                        if vis_input == 'y':
                            chart_agent = ChartAgent(df)
                            chart_agent.plot(question)
                    except Exception as e:
                        print(f"Could not generate chart: {e}")

                # Feedback Loop (only if not from memory)
                if not from_memory:
                    feedback = input("\nWas this correct? (y/n): ").strip().lower()
                    if feedback == 'y':
                        save_memory(question, sql_query)
            else:
                print("Query returned no results.")

        except Exception as e:
            print(f"Failed to execute query: {e}")

if __name__ == "__main__":
    main()

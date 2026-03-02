import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine


# ---------------- DB CONNECTION ----------------
def get_db_engine():
    connection_string = (
        "mssql+pyodbc://localhost/Project?"
        "driver=ODBC+Driver+17+for+SQL+Server&"
        "trusted_connection=yes&"
        "TrustServerCertificate=yes"
    )
    return create_engine(connection_string)


engine = get_db_engine()

query = """
SELECT TOP 10 Id, Title, Description
FROM dbo.azure_devops_bugs
ORDER BY Id DESC;

"""

df = pd.read_sql(query, engine)



# ---------------- CATEGORIES (REQUIRED) ----------------
Categories = [
    "Login Issues",
    "UI Issues",
    "Backend Issues",
    "Database Errors",
    "Feature Requests",
    "Uncategorized"
]


# ---------------- LLM ----------------
llm = ChatOllama(
    model="mistral:7b",
    temperature=0,
    stream=False   # IMPORTANT
)


# ---------------- PROMPT ----------------
prompt = PromptTemplate(
    input_variables=["text", "categories"],
    template="""
You are a strict text classifier.

Choose EXACTLY ONE category from the list below.
If none fit, choose "Uncategorized".

Categories:
{categories}

Text:
{text}

Return ONLY the category name.
"""
)


chain = prompt | llm | StrOutputParser()


# ---------------- CLASSIFIER ----------------
def classify_text_column(df):
    categories_str = ", ".join(Categories)
    results = []

    for _, row in df.iterrows():
        desc = str(row.get("Description", ""))[:300]  # truncate for speed

        text = f"""
Title: {row.get('Title', '')}
Description: {desc}
Tags: {row.get('Tags', '')}
""".strip()

        category = chain.invoke({
            "text": text,
            "categories": categories_str
        }).strip()

        if category not in Categories:
            category = "Uncategorized"

        results.append(category)

    return results


# ---------------- RUN ----------------
df["Category"] = classify_text_column(df)
print(df[["Id", "Title", "Category"]].head())

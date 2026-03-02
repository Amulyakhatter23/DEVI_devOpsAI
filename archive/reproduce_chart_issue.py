
import pandas as pd
from chart_agent import ChartAgent

# Mock Data simulating the user's query result
data = {
    "State": ["Active", "Closed", "New", "Resolved"],
    "CountOfTickets": [10, 50, 5, 20]
}
df = pd.DataFrame(data)

print("DataFrame:")
print(df)

agent = ChartAgent(df)
question = "Visualize the count of tickets by state"

print("\nGenerating Code...")
# We need to access the text output, not execute it immediately to see what's wrong
# Accessing hidden method or just calling plot which prints the code? 
# ChartAgent.plot executes internal logic. 
# I will modify ChartAgent temporarily or just use the chain directly if accessible.
# ChartAgent has self.chain.

numeric_cols = df.select_dtypes(include="number").columns.tolist()
datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
categorical_cols = df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()

response = agent.chain.invoke({
    "question": question,
    "numeric_cols": numeric_cols,
    "datetime_cols": datetime_cols,
    "categorical_cols": categorical_cols,
    "dtypes": str(df.dtypes),
    "head": df.head().to_string()
})

with open("debug_chart_code.txt", "w", encoding="utf-8") as f:
    f.write(response)

print("Code written to debug_chart_code.txt")

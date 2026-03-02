import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_sql import get_app_components, save_memory, load_memory
from chart_agent import ChartAgent
import re

def generate_chart_code_manually(chart_agent, question, df):
    """
    Manually invokes the chart agent chain to get plot code.
    This is a workaround because ChartAgent was reverted to an older version
    that doesn't expose get_plot_code.
    """
    print("Streamlit: Generating chart code manually...")
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()
    dtypes = str(df.dtypes)
    try:
        head = df.head().to_string()
    except ImportError:
        head = str(df.head())
    
    plot_code = chart_agent.chain.invoke({
        "question": question,
        "numeric_cols": numeric_cols,
        "datetime_cols": datetime_cols,
        "categorical_cols": categorical_cols,
        "dtypes": dtypes,
        "head": head
    })
    
    # Clean code logic
    match = re.search(r"```.*?\n(.*?)```", plot_code, flags=re.S)
    if match:
        plot_code = match.group(1).strip()
    else:
        plot_code = plot_code.strip()
        if plot_code.startswith("```python"): plot_code = plot_code[9:]
        if plot_code.startswith("```"): plot_code = plot_code[3:]
        if plot_code.endswith("```"): plot_code = plot_code[:-3]
        plot_code = plot_code.strip()
        
    return plot_code

def main():
    st.title("SQL Query Assistant")

    # Initialize components
    if "components" not in st.session_state:
        with st.spinner("Connecting to database..."):
            st.session_state.components = get_app_components()

    if not st.session_state.components:
        st.error("Failed to initialize application components. Check database connection.")
        st.stop()

    components = st.session_state.components
    engine = components["engine"]
    generate_query = components["generate_query"]
    review_chain = components["review_chain"]
    extract_sql = components["extract_sql"]

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "df" in message:
                st.dataframe(message["df"])
            if "chart_code" in message:
                try:
                    exec_globals = {'df': message["df"], 'plt': plt, 'pd': pd}
                    exec(message["chart_code"], exec_globals)
                    st.pyplot(plt)
                    plt.clf() # Clear figure
                except Exception as e:
                    st.error(f"Error displaying chart: {e}")
            
            # Feedback for SQL
            if "sql_query" in message and not message.get("from_memory", False):
                if st.button("Save Query to Memory", key=f"save_{st.session_state.messages.index(message)}"):
                    save_memory(message["question"], message["sql_query"])
                    st.success("Query saved to memory!")
                    message["from_memory"] = True
                    st.rerun()

    # User Input
    if prompt := st.chat_input("Ask a question about Azure DevOps bugs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                memory = load_memory()
                sql_query = ""
                from_memory = False
                
                if prompt in memory:
                    sql_query = memory[prompt]
                    from_memory = True
                    message_placeholder.markdown(f"**Retrieved from memory:**\n```sql\n{sql_query}\n```")
                else:
                    message_placeholder.markdown("Generating SQL query...")
                    response = generate_query.invoke({"input": prompt})
                    sql_query = response.strip()
                    if sql_query.startswith("```sql"): sql_query = sql_query[6:]
                    if sql_query.endswith("```"): sql_query = sql_query[:-3]
                    sql_query = sql_query.strip()
                    
                    message_placeholder.markdown(f"Reviewing SQL...\n```sql\n{sql_query}\n```")
                    reviewed_sql = review_chain.invoke({"input": prompt, "sql_query": sql_query})
                    sql_query = extract_sql(reviewed_sql)
                    
                    message_placeholder.markdown(f"**Executed SQL:**\n```sql\n{sql_query}\n```")

                df = pd.read_sql(sql_query, engine)
                
                if not df.empty:
                    st.dataframe(df)
                    
                    assistant_msg = {
                        "role": "assistant", 
                        "content": f"**Executed SQL:**\n```sql\n{sql_query}\n```", 
                        "df": df,
                        "sql_query": sql_query,
                        "question": prompt,
                        "from_memory": from_memory
                    }
                    st.session_state.messages.append(assistant_msg)

                else:
                    st.warning("No results found.")
                    st.session_state.messages.append({"role": "assistant", "content": "No results found."})

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

    # Visualization Section
    if st.session_state.messages and "df" in st.session_state.messages[-1]:
        last_msg = st.session_state.messages[-1]
        if st.button("Visualize Last Result"):
            with st.spinner("Generating chart..."):
                try:
                    chart_agent = ChartAgent(last_msg["df"])
                    question = st.session_state.messages[-2]["content"]
                    
                    plot_code = generate_chart_code_manually(chart_agent, question, last_msg["df"])
                    
                    exec_globals = {'df': last_msg["df"], 'plt': plt, 'pd': pd}
                    exec(plot_code, exec_globals)
                    st.pyplot(plt)
                    
                    last_msg["chart_code"] = plot_code
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Could not generate chart: {e}")

if __name__ == "__main__":
    st.set_page_config(page_title="SQL Assistant", page_icon=None, layout="wide")
    main()

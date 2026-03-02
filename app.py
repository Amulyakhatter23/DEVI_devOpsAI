import streamlit as st
import pandas as pd
import json
import os
import data_loader
import config
import streamlit_app
import numpy as np
import plotly.express as px
from ollama_service import ollama_service



def find_similar_tickets(query_text, ticket_embeddings, tickets_df, top_k=5, model_name=config.OLLAMA_MODEL):
    """
    Find similar tickets based on cosine similarity.
    """
    # Use centralized service for embeddings (Cached)
    try:
        query_embedding = ollama_service.get_embeddings([query_text], model_name=model_name)[0]
    except Exception as e:
        st.error(f"Error generating embedding for query: {e}")
        return pd.DataFrame()

    # Calculate Cosine Similarity
    # Sim(A, B) = (A . B) / (||A|| * ||B||)
    
    similarities = []
    query_norm = np.linalg.norm(query_embedding)
    
    if query_norm == 0:
        return pd.DataFrame()

    for i, emb in enumerate(ticket_embeddings):
        if len(emb) == 0:
            similarities.append(0)
            continue
            
        emb_norm = np.linalg.norm(emb)
        if emb_norm == 0:
            similarities.append(0)
            continue
            
        sim = np.dot(query_embedding, emb) / (query_norm * emb_norm)
        similarities.append(sim)
        
    # Add similarity scores to DataFrame
    results_df = tickets_df.copy()
    results_df['Similarity'] = similarities
    
    # Sort by similarity
    results_df = results_df.sort_values(by='Similarity', ascending=False)
    
    # Filter out the query itself (if it exists in the list) and return top_k
    # We assume exact title match means it's the same ticket
    results_df = results_df[results_df['Title'] != query_text]
    
    return results_df.head(top_k)
# ----------------------------------------------

# Set page config globally
st.set_page_config(page_title="DEVI(DevOps AI)", page_icon="🚀", layout="wide")

# ---------------- OLLAMA LOGIC ----------------

# ----------------------------------------------

def run_ticket_classifier():
    st.title("Devi(DevOps AI)")

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    
    # AI Settings
    st.sidebar.header("AI Settings")
    ollama_model = st.sidebar.text_input("Ollama Model", value="qwen2.5-coder:7b")
    
    # # Project Settings
    # project_name = st.sidebar.text_input("Project Name", value="MCP_ChangeLog")
    # org_url = st.sidebar.text_input("Organization URL", value="https://dev.azure.com/Amulyakhatter23")
    
    # # Date Filter
    # start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2025-12-01"))
    
    # Ticket Limit
    limit = st.sidebar.number_input("Max Tickets to Process", min_value=1, max_value=2000, value=200, step=50)

    # --- Main Content ---
    
    st.subheader("🧠 Knowledge Base (Semantic Rules)")
    st.info("Define categories and their **descriptions**. The AI will use the meaning to classify tickets.")
    
    # Default Knowledge Base (Semantic)
    default_kb = {"Login & Access Bugs": "Bugs related to authentication and authorization such as login failures, password issues, SSO problems, access denial, and Home, Landing Page, or Search access failures.",
  "UI/UX Bugs": "Visual and usability bugs including layout breaks, CSS issues, responsiveness problems, font or color mismatches, and interaction defects across web and mobile.",
  "Backend & Database Bugs": "Server-side bugs including API failures, business logic errors, performance issues, database connection problems, SQL errors, timeouts, and data inconsistencies.",
  "Analytics & Reporting Bugs": "Bugs related to GA4, GTM, event tracking failures, missing or duplicate data, incorrect metrics, and reporting discrepancies.",
  "Finance & Payment Bugs": "Payment-related bugs including transaction failures, incorrect refunds, invoice issues, SAP feed errors, credit limit problems, and reconciliation mismatches.",
  "Onboarding & Profile Bugs": "Bugs in onboarding and profile flows including KYC failures, profile update errors, account activation/deactivation issues for buyers or sellers.",
  "Mobile Buyer-Side Bugs": "Bugs specific to the buyer mobile application including crashes, broken flows, performance issues, and mobile-only feature defects.",
  "Mobile Seller-Side Bugs": "Bugs specific to the seller mobile application including listing issues, order handling bugs, notification failures, and seller workflow errors.",
  "Order Management Bugs": "Bugs related to order lifecycle such as order creation failures, incorrect status updates, cancellation issues, fulfillment errors, and order tracking problems.",
  "RFQ Bugs": "Bugs in RFQ workflows including RFQ creation failures, supplier response issues, negotiation errors, pricing mismatches, and status update problems.",
  "Catalogue Bugs": "Bugs related to product catalogue management including incorrect listings, pricing errors, missing images, categorization issues, and search indexing failures.",
  "Logistics Bugs": "Bugs related to shipping and logistics including delivery tracking errors, dispatch issues, partner integration failures, returns, and shipment status mismatches.",
  "AI & Automation Bugs": "Bugs in AI-driven features such as incorrect predictions, failed auto-categorization, broken recommendations, automation misfires, or ML output errors."
}
    
    
    kb_input = st.text_area("Input JSON Rules", value=json.dumps(default_kb, indent=4), height=300)
    
    try:
        keyword_rules = json.loads(kb_input)
    except json.JSONDecodeError:
        st.error("Invalid JSON format in Knowledge Base.")
        return

    # 2. Run Analysis
    
    # Data Source Selection
    st.subheader("Data Source: SQL Server")
    data_source = "SQL Server"

    #Add a Sample Head view of the datasource.

    default_conn = config.DB_CONNECTION_STRING
    sql_conn_str = st.text_input("Connection String", value=default_conn)
    col1, col2 = st.columns(2)
    sql_table = col1.text_input("Table Name", value=config.TABLE_NAME)
    sql_schema = col2.text_input("Schema", value=config.SCHEMA_NAME)

    if st.button("Run Analysis", type="primary"):
        # Validation
        if not sql_conn_str or not sql_table:
            st.error("Please provide Connection String and Table Name.")
            return

        status_container = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # A. Fetch Data
            df = pd.DataFrame()
            
            status_container.write(f"Fetching from SQL Server ({sql_schema}.{sql_table})...")
            items = data_loader.fetch_from_sql(sql_conn_str, sql_table, sql_schema)
            df = pd.DataFrame(items).head(limit)
                
            # Normalize columns
            if 'Title' not in df.columns:
                st.error(f"Data must have a 'Title' column. Found: {list(df.columns)}")
                return
            
            # Fill missing standard columns
            if 'Id' not in df.columns:
                df['Id'] = range(1, len(df) + 1)
                df['Id'] = df['Id'].astype(str)
            if 'Description' not in df.columns:
                df['Description'] = ""
            if 'Tags' not in df.columns:
                df['Tags'] = ""
            if 'AreaPath' not in df.columns:
                df['AreaPath'] = ""
            if 'State' not in df.columns:
                df['State'] = "Active"
            if 'AssignedTo' not in df.columns:
                df['AssignedTo'] = "Unassigned"
            if 'Created Date' not in df.columns:
                df['Created Date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # Ensure Id is string
            df['Id'] = df['Id'].astype(str)

            if df.empty:
                st.warning("No tickets found.")
                return
                
            progress_bar.progress(20)
            status_container.write(f"Fetched {limit} tickets. Preparing for classification...")
            
            # B. Prepare Data
            # Initialize Category and Confidence
            df['Category'] = 'Uncategorized'
            df['Confidence'] = 0
            
            # Limit tickets
            tickets_to_process = df[['Id', 'Title', 'Description', 'Tags', 'AreaPath']].head(limit).to_dict('records')
            
            # C. Classification
            status_container.write(f"Classifying {len(tickets_to_process)} tickets using Ollama ({ollama_model})...")
            
            # Ollama Classification
            results = ollama_service.classify_tickets(tickets_to_process, keyword_rules, ollama_model)
            
            if "error" in results:
                st.error(f"Classification Failed: {results['error']}")
                st.warning("Please check if Ollama is running (`ollama serve`).")
                return

            progress_bar.progress(80)
            status_container.write("Updating results...")
            
            # Update DataFrame
            for cat, items in results.items():
                for item in items:
                    df.loc[df['Id'] == item['Id'], 'Category'] = cat
                    df.loc[df['Id'] == item['Id'], 'Confidence'] = item['Confidence']
            
            progress_bar.progress(100)
            status_container.success("Analysis Complete!")
            
            # Store in Session State
            st.session_state['df_results'] = df
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

    # 3. Results Display (Outside Button Logic)
    if 'df_results' in st.session_state:
        df = st.session_state['df_results']
        
        st.divider()
        
        # --- Filters ---
        st.subheader("🔍 Filter Results")
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            state_options = ["All"] + sorted(df['State'].astype(str).unique())
            state_filter = st.selectbox("State", options=state_options, index=0)
        with f_col2:
            assignee_options = ["All"] + sorted(df['AssignedTo'].astype(str).unique())
            assignee_filter = st.selectbox("Assigned To", options=assignee_options, index=0)
        # with f_col3:
        #     area_options = ["All"] + sorted(df['AreaPath'].astype(str).unique())
        #     area_filter = st.selectbox("Area Path", options=area_options, index=0)
            
        # Apply Filters
        df_filtered = df.copy()
        
        if state_filter != "All":
            df_filtered = df_filtered[df_filtered['State'] == state_filter]
            
        if assignee_filter != "All":
            df_filtered = df_filtered[df_filtered['AssignedTo'] == assignee_filter]
            
        # if area_filter != "All":
        #     df_filtered = df_filtered[df_filtered['AreaPath'] == area_filter]
        
        if df_filtered.empty:
            st.warning("No tickets match the selected filters.")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tickets", len(df_filtered))
        col2.metric("Classified", len(df_filtered[df_filtered['Category'] != 'Uncategorized']))
        col3.metric("Uncategorized", len(df_filtered[df_filtered['Category'] == 'Uncategorized']))
        
        # Charts
        st.subheader("Category Distribution")
        if not df_filtered.empty:
            counts = df_filtered['Category'].value_counts().reset_index()
            counts.columns = ['Category', 'Count']
            
            fig = px.bar(counts, x='Category', y='Count', color='Category', title="Tickets by Category")

            
            # Interactive Chart
            selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
            
            # Handle Selection
            selected_category = None
            if selection and len(selection["selection"]["points"]) > 0:
                selected_category = selection["selection"]["points"][0]["x"]
                st.info(f"Filtering for Category: **{selected_category}**")
                df_filtered = df_filtered[df_filtered['Category'] == selected_category]
        
        # Data Table
        st.subheader("Ticket Details")
        st.dataframe(df_filtered[['Id', 'Title', 'State', 'AssignedTo', 'Created Date','Category', 'Confidence','Changed_Date']])
        
        # Download
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name='ticket_analysis_results.csv',
            mime='text/csv',
        )

        # --- Similarity Search ---
        st.divider()
        st.subheader("🔎 Find Similar Tickets")
        
        search_mode = st.radio("Search By:", ["Select Existing Ticket", "Custom Query"], horizontal=True)
        
        query_text = ""
        if search_mode == "Select Existing Ticket":
            selected_ticket_title = st.selectbox("Select a Ticket", options=df['Title'].unique())
            if selected_ticket_title:
                query_text = selected_ticket_title
        else:
            query_text = st.text_input("Enter Issue Description or Title")
            
        if st.button("Find Similar"):
            if not query_text:
                st.warning("Please enter a query or select a ticket.")
            else:
                with st.spinner("Generating embeddings and searching..."):
                    # Generate embeddings for ALL tickets (cached)
                    # We use Title + Tags for better context
                    text_data = (df['Title'] + " " + df['Tags']).tolist()
                    ticket_embeddings = ollama_service.get_embeddings(text_data, ollama_model)
                    
                    if len(ticket_embeddings) > 0:
                        similar_df = find_similar_tickets(query_text, ticket_embeddings, df, top_k=5, model_name=ollama_model)
                        
                        if not similar_df.empty:
                            st.success(f"Found {len(similar_df)} similar tickets:")
                            # Display as a table with similarity bar
                            st.dataframe(
                                similar_df[['Id', 'Title', 'State', 'AssignedTo', 'Similarity']],
                                column_config={
                                    "Similarity": st.column_config.ProgressColumn(
                                        "Similarity Score",
                                        help="Cosine similarity (0-1)",
                                        format="%.2f",
                                        min_value=0,
                                        max_value=1,
                                    ),
                                }
                            )
                        else:
                            st.info("No similar tickets found.")
                    else:
                        st.error("Failed to generate embeddings.")

def main():
    st.sidebar.title("🚀 Navigation")
    
    # Default to Ticket Classifier
    app_mode = st.sidebar.radio(
        "Choose Application:",
        ["Ticket Classifier", "SQL Assistant"]
    )
    
    st.sidebar.markdown("---")
    
    if app_mode == "Ticket Classifier":
        run_ticket_classifier()
    elif app_mode == "SQL Assistant":
        streamlit_app.main()

if __name__ == "__main__":
    main()

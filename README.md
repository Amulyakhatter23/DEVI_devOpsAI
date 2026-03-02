# DEVI - DevOps AI 🚀

DEVI (DevOps AI) is an intelligent assistant designed to streamline Azure DevOps workflows by classifying bug tickets and providing a natural language interface for SQL queries.

## 🌟 Features

- **Ticket Classifier**: Automatically categorizes Azure DevOps bug tickets using Ollama-powered Large Language Models (LLMs). It uses semantic rules to assign categories like "Login & Access Bugs", "UI/UX Bugs", and more.
- **SQL Assistant**: A conversational interface that allows users to ask questions about their database in plain English. It generates and executes SQL queries, then visualizes the results.
- **Similarity Search**: Find related tickets based on semantic similarity using embeddings.
- **Interactive Dashboards**: Built with Streamlit for a seamless and responsive user experience.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **AI/ML**: [Ollama](https://ollama.com/) (Qwen2.5-Coder), [LangChain](https://www.langchain.com/)
- **Data**: [Pandas](https://pandas.pydata.org/), [SQL Server](https://www.microsoft.com/en-us/sql-server)
- **Visualization**: [Matplotlib](https://matplotlib.org/), [Plotly](https://plotly.com/)

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running (`ollama serve`).
- A running SQL Server instance with the required database.
- ODBC Driver 17 for SQL Server.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Amulyakhatter23/Final_azureDevOps_Last.git
    cd "DEVI - DevOps AI"
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the database**:
    Update `config.py` with your SQL Server connection details.

4.  **Pull the LLM model**:
    ```bash
    ollama pull qwen2.5-coder:7b
    ```

## 📈 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

- Use the **Navigation** sidebar to switch between the "Ticket Classifier" and "SQL Assistant".
- In the **Ticket Classifier**, define your semantic rules in JSON format and run analysis.
- In the **SQL Assistant**, type your questions in the chat input to query the database.

## 📝 License

This project is licensed under the MIT License.

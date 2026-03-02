
import config
import ollama_service
import langchain_sql
import json
import numpy as np
import time

def test_classification():
    print("\n-------------------------------------------------")
    print("TEST [TC-01, TC-04]: Classification & Confidence")
    print("-------------------------------------------------")
    
    tickets = [
        {"Id": 1, "Title": "Login page is throwing 500 error", "Tags": "Web, Critical", "AreaPath": "Project\\Web"},
        {"Id": 2, "Title": "Add dark mode to settings", "Tags": "UI", "AreaPath": "Project\\UI"},
        {"Id": 3, "Title": "Database query performance slow", "Tags": "DB, Perf", "AreaPath": "Project\\DB"},
    ]
    
    categories = {
        "Bug": "Issues related to functionality errors",
        "Feature": "New capabilities or improvements",
        "UI/UX": "Visual design and user experience",
        "Database": "SQL, Performance, Data integrity"
    }
    
    print(f"Testing classification for {len(tickets)} tickets...")
    try:
        results = ollama_service.ollama_service.classify_tickets(tickets, categories)
        
        # Verify structure
        if not isinstance(results, dict):
            print("2. [FAIL]: Result is not a dictionary")
            return

        total_classified = sum(len(v) for v in results.values())
        if total_classified != len(tickets):
            print(f"2. [FAIL]: Expected {len(tickets)} classifications, got {total_classified}")
        else:
            print(f"1. [PASS]: Classified {total_classified} tickets.")
            
        # Check Confidence
        for cat, items in results.items():
            for item in items:
                print(f"   ID: {item['Id']} -> Category: {cat} | Conf: {item.get('Confidence', 'N/A')}")
                if 'Confidence' not in item or not isinstance(item['Confidence'], int):
                     print(f"   [WARN]: Invalid Confidence for ID {item['Id']}")

    except Exception as e:
        print(f"2. [FAIL]: Exception during classification: {e}")

def test_embeddings():
    print("\n-------------------------------------------------")
    print("TEST [TC-05]: Embedding Generation")
    print("-------------------------------------------------")
    
    text = ["This is a test ticket for embedding generation."]
    try:
        embeddings = ollama_service.ollama_service.get_embeddings(text)
        
        if isinstance(embeddings, np.ndarray) and embeddings.shape[0] == 1:
             print(f"1. [PASS]: Generated embedding vector of length {embeddings.shape[1]}")
        else:
             print(f"2. [FAIL]: Invalid embedding format or shape: {embeddings.shape}")
             
    except Exception as e:
        print(f"2. [FAIL]: Exception during embedding: {e}")

def test_sql_generation():
    print("\n-------------------------------------------------")
    print("TEST [TC-06, TC-07]: SQL Generation (RAG)")
    print("-------------------------------------------------")
    
    components = langchain_sql.get_app_components()
    if not components:
        print("2. [FAIL]: Could not initialize app components (DB Connection issues?)")
        return

    generate_query = components["generate_query"]
    
    test_cases = [
        {"id": "TC-06", "q": "Show me all active bugs", "expect": "State"},
        {"id": "TC-07", "q": "Bugs created last month", "expect": "DATEDIFF"}, 
    ]
    
    for tc in test_cases:
        print(f"\nTesting: '{tc['q']}'")
        try:
            response = generate_query.invoke({"input": tc['q']})
            extract_sql_fn = components["extract_sql"]
            sql = extract_sql_fn(response)
            
            print(f"Generated SQL: {sql[:50]}...")
            
            if tc['expect'] in sql:
                print(f"1. [PASS] {tc['id']}")
            else:
                 print(f"3. [WARN] {tc['id']}: Expected keyword '{tc['expect']}' not found.")
                 
        except Exception as e:
            print(f"2. [FAIL] {tc['id']}: Exception: {e}")

def test_performance():
    import time
    print("\n-------------------------------------------------")
    print("TEST [2.3]: Performance Evaluation (Latency)")
    print("-------------------------------------------------")
    
    # 1. Classification Latency
    tickets = [{"Id": i, "Title": f"Ticket {i}", "Tags": "", "AreaPath": ""} for i in range(1, 6)] # 5 tickets
    categories = {"Bug": "Issues", "Feature": "New things"}
    
    start_time = time.time()
    ollama_service.ollama_service.classify_tickets(tickets, categories)
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / 5
    print(f"4. [METRICS] Classification (5 tickets): Total={total_time:.2f}s | Avg={avg_time:.2f}s/ticket")

    # 2. Embedding Latency
    start_time = time.time()
    ollama_service.ollama_service.get_embeddings(["Performance test query string"])
    end_time = time.time()
    print(f"4. [METRICS] Embedding Generation: {(end_time - start_time):.4f}s")
    
    # 3. SQL Generation Latency
    components = langchain_sql.get_app_components()
    if components:
        generate_query = components["generate_query"]
        start_time = time.time()
        try:
            generate_query.invoke({"input": "Show me all active bugs"})
            end_time = time.time()
            print(f"4. [METRICS] SQL Generation: {(end_time - start_time):.2f}s")
        except:
            print("4. [METRICS] SQL Generation: Failed (Time not measured)")

if __name__ == "__main__":
    print(">>> STARTING AUTOMATED TEST RUNNER (Integration Tests)")
    print(f"Ollama Model: {config.OLLAMA_MODEL}")
    
    test_classification()
    test_embeddings()
    test_sql_generation()
    test_performance()
    
    print("\n>>> TEST RUN COMPLETE")

import ollama
import json
import numpy as np
import streamlit as st
from langchain_ollama import ChatOllama
import config

class OllamaService:
    def __init__(self):
        self.host = 'http://127.0.0.1:11434'
        self.client = ollama.Client(host=self.host)
        self.default_model = config.OLLAMA_MODEL

    def get_client(self):
        """Returns the raw Ollama client."""
        return self.client

    def get_chat_model(self, temperature=0):
        """Returns a LangChain ChatOllama instance."""
        return ChatOllama(model=self.default_model, temperature=temperature)

    @st.cache_data
    def get_embeddings(_self, text_list, model_name=None):
        """
        Generate embeddings for a list of texts using Ollama.
        Cached to avoid re-computation.
        Note: _self is used to exclude self from caching hash.
        """
        if model_name is None:
            model_name = _self.default_model

        embeddings = []
        
        for text in text_list:
            try:
                response = _self.client.embeddings(model=model_name, prompt=text)
                embeddings.append(response["embedding"])
            except Exception as e:
                print(f"Error getting embedding: {e}")
                embeddings.append([]) # Handle error gracefully
                
        return np.array(embeddings)

    def classify_tickets(self, tickets, categories, model_name=None):
        """
        Classifies a list of tickets into categories.
        
        tickets: list of dicts with Id, Title, Tags, AreaPath
        categories: { "Category": "Description" }
        
        returns: { category: [ {Id, Confidence} ] }
        """
        if model_name is None:
            model_name = self.default_model

        from collections import defaultdict
        final_results = defaultdict(list)
        
        # Format categories for Prompt
        category_text = json.dumps(categories, indent=2)

        # Process ALL tickets with LLM (Semantic Matching)
        for t in tickets:
            prompt = f'''You are a Ticket Classifier.
Analyze the following ticket against ALL provided categories.

### Categories & Definitions
{category_text}

### Ticket
Title: {t['Title']}
Tags: {t.get('Tags','')}
Area Path: {t.get('AreaPath','')}

### Instructions
1. Assign a relevance score between 0.0 and 1.0 to EACH category.
2. Return JSON ONLY in this exact format:
{{
  "{t['Id']}": {{
    "scores": {{
      "Category1": 0.85,
      "Category2": 0.10,
      "Uncategorized": 0.05
    }}
  }}
}}
'''

            try:
                response = self.client.chat(
                    model=model_name,
                    options={"temperature": 0.0},
                    messages=[{"role": "user", "content": prompt}]
                )

                content = response["message"]["content"]
                if "```" in content:
                    content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                
                # Logic: Competitive Confidence
                result_data = list(data.values())[0]
                
                if isinstance(result_data, dict) and "scores" in result_data:
                    scores = result_data["scores"]
                    
                    if not scores:
                        best_cat = "Uncategorized"
                        conf = 0
                    else:
                        # Find winner
                        best_cat = max(scores, key=scores.get)
                        max_score = scores[best_cat]
                        total_score = sum(scores.values())
                        
                        # Compute confidence as share of total probability
                        if total_score > 0:
                            conf = int((max_score / total_score) * 100)
                        else:
                            conf = 0
                else:
                    # Fallback
                    best_cat = "Uncategorized"
                    conf = 0

                if best_cat not in categories and best_cat != "Uncategorized":
                    best_cat = "Uncategorized"
                    conf = 0

                final_results[best_cat].append({"Id": t["Id"], "Confidence": conf})

            except Exception as e:
                print(f"Error classifying ticket {t['Id']}: {e}")
                final_results["Uncategorized"].append({"Id": t["Id"], "Confidence": 0})

        return dict(final_results)

# Singleton instance
ollama_service = OllamaService()

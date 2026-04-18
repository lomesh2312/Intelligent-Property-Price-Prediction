import os
from langchain_groq import ChatGroq

class RAGEngine:
    def __init__(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.knowledge_context = f.read()
        except Exception as e:
            self.knowledge_context = ""
            print(f"Warning: Could not read knowledge base. {e}")

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

    def query(self, query):
        prompt = f"""
You are a legal property assistant.

STRICT RULES:
- Give SHORT and PRECISE answers
- ONLY answer what is asked
- Do NOT include:
  - market trends
  - recommendations
  - risks
- Use bullet points if needed
- Stick to legal facts only
- Under NO circumstances should you use Markdown asterisks (**) for bolding.

Context:
{self.knowledge_context}

Question:
{query}

Answer:
"""
        response = self.llm.invoke(prompt)
        return response.content.strip()

# Initialize an instance
current_dir = os.path.dirname(__file__)
knowledge_file_path = os.path.join(current_dir, "data", "real_estate_knowledge.txt")
try:
    rag = RAGEngine(knowledge_file_path)
except Exception as e:
    print(f"Warning: Failed to load RAG engine. {e}")
    rag = None

import streamlit as st
from llama_stack_client import LlamaStackClient

# Initialize client (adjust base_url if needed)
client = LlamaStackClient(base_url="http://lsd-llama-milvus-service.rag.svc.cluster.local:8321")

vector_db_id = "my-milvus-db"

st.set_page_config(page_title="Conference Concierge", page_icon="🤖")
st.title("🎤 Conference Concierge Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
user_input = st.chat_input("Ask me about TIP25 sessions...")
if user_input:
    # Query your RAG tool
    result = client.tool_runtime.rag_tool.query(
        vector_db_ids=[vector_db_id],
        content=user_input,
    )

    # For simplicity, just take the top result’s text
    answer = ""
    if result.metadata.get("chunks"):
        answer = result.metadata["chunks"][0]
    else:
        answer = "I couldn’t find anything relevant."

    # Save to history
    st.session_state.history.append((user_input, answer))

# Display chat history
for q, a in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)
import os

import streamlit as st
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent

# Initialize client
base_url = os.getenv("LLAMA_STACK_URL", "http://lsd-llama-milvus-inline-service.rag.svc.cluster.local:8321")
print(f"Using base URL: {base_url}")
client = LlamaStackClient(base_url=base_url)
dbs = client.vector_dbs.list()
identifier = dbs[0].identifier
vector_db_id = identifier

model_id = "llama-model"

st.set_page_config(page_title="Conference Concierge", page_icon="🤖")
st.title("🎤 Conference Concierge Chatbot")

# Add connection test in sidebar
with st.sidebar:
    st.header("Connection Status")

    # Test connection
    try:
        models = client.models.list()
        st.success(f"✅ Connected to Llama Stack")
        st.write(f"Available models: {len(models)}")
        model_id = models[0].identifier if models else "meta-llama/Llama-3.2-3B-Instruct"
        st.write(f"Using model: {model_id}")
    except Exception as e:
        st.error(f"❌ Connection failed: {str(e)}""")
        st.write(f"Server URL: {client.base_url}")
        model_id = None

# Only proceed if we have a connection
if model_id is None:
    st.error("Cannot connect to Llama Stack server. Please check:")
    st.markdown("""
    1. Is the server URL correct?
    2. Is the Llama Stack service running?
    3. Can your app reach the service (network/firewall)?
    4. Try: `kubectl get pods -n rag` to check if pods are running
    """)
    st.stop()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "agent" not in st.session_state:
    try:
        # Create an Agent for conversational RAG queries - matching your example format exactly
        st.session_state.agent = Agent(
            client,
            model=model_id,
            instructions="""You are a helpful conference concierge assistant.
            When users ask about sessions, provide a friendly, conversational response (2-3 sentences).
            Summarize what you found and highlight the most relevant information.
            Keep it concise - detailed session information will be shown in a table.""",
            tools=[
                {
                    "name": "builtin::rag/knowledge_search",
                    "args": {"vector_db_ids": [vector_db_id]},
                }
            ],
        )
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        st.stop()
if "session_id" not in st.session_state:
    try:
        st.session_state.session_id = st.session_state.agent.create_session("conference_chat")
    except Exception as e:
        st.error(f"Error creating session: {str(e)}")
        st.stop()


# Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.session_state.session_id = st.session_state.agent.create_session("conference_chat")
        st.rerun()

# Chat input
user_input = st.chat_input("Ask me about the conference...")
if user_input:
    with st.spinner("Searching and generating response..."):

        # Get conversational response from agent (streaming enabled)
        response = st.session_state.agent.create_turn(
            messages=[{"role": "user", "content": user_input}],
            session_id=st.session_state.session_id,
            stream=True,
        )

        # Collect the streamed response
        full_response = ""
        for event in response:
            if hasattr(event, "event") and hasattr(event.event, "payload"):
                payload = event.event.payload

                # Handle streaming chunks (preferred path)
                if hasattr(payload, "delta") and hasattr(payload.delta, "content"):
                    chunk = payload.delta.content
                    full_response += chunk
                    # Optionally show live updates in the UI
                    st.write(chunk)

                # Handle final completed message (end of stream)
                elif hasattr(payload, "turn") and hasattr(payload.turn, "output_message"):
                    if hasattr(payload.turn.output_message, "content"):
                        # Only append if not already captured by deltas
                        if not full_response:
                            full_response = payload.turn.output_message.content

        # Save to history
        st.session_state.history.append({
            "query": user_input,
            "response": full_response if full_response else "I found some relevant sessions for you."
        })

# Display chat history
for item in st.session_state.history:
    st.chat_message("user").write(item["query"])
    with st.chat_message("assistant"):
        st.write(item["response"])

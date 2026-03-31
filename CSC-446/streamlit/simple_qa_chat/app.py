import os
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Simple QA Chat", page_icon="💬")
st.title("💬 Simple QA Chat")

# API key from sidebar
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful assistant. Answer questions clearly and concisely.",
    )

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages for API call
    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(st.session_state.messages)

    # Get response
    with st.chat_message("assistant"):
        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})

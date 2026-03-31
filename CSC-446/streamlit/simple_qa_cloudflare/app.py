import os
import requests
import streamlit as st

st.set_page_config(page_title="QA Chat (Cloudflare AI)", page_icon="☁️")
st.title("☁️ QA Chat with Cloudflare Workers AI")

with st.sidebar:
    account_id = st.text_input("Cloudflare Account ID", type="password", value=os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
    api_token = st.text_input("Cloudflare API Token", type="password", value=os.getenv("CLOUDFLARE_API_TOKEN", ""))
    model_name = st.selectbox(
        "Model",
        [
            "@cf/meta/llama-3.1-8b-instruct",
            "@cf/mistralai/mistral-7b-instruct-v0.1",
            "@cf/meta/llama-2-7b-chat-fp16",
            "@cf/thebloke/zephyr-7b-beta-awq",
            "@cf/tiiuae/falcon-7b-instruct",
        ],
        index=0,
    )
    max_tokens = st.slider("Max tokens", 64, 1024, 256)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7)
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful assistant. Answer questions clearly and concisely.",
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    if not account_id or not api_token:
        st.error("Please enter your Cloudflare Account ID and API Token in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages for API call
    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(st.session_state.messages)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}"
            headers = {"Authorization": f"Bearer {api_token}"}
            payload = {
                "messages": api_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            resp = requests.post(url, headers=headers, json=payload)

            if resp.status_code != 200:
                st.error(f"API error ({resp.status_code}): {resp.text}")
                st.stop()

            data = resp.json()
            if not data.get("success"):
                errors = data.get("errors", [])
                st.error(f"Cloudflare API error: {errors}")
                st.stop()

            response = data["result"]["response"]
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

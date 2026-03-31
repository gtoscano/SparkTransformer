import os
import warnings

# Suppress transformers/torch warnings before any imports
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import streamlit as st
from transformers import pipeline, GenerationConfig

st.set_page_config(page_title="QA Chat (HuggingFace)", page_icon="🤗")
st.title("🤗 QA Chat with HuggingFace Transformers")

with st.sidebar:
    model_name = st.selectbox(
        "Model",
        [
            "distilgpt2",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/phi-2",
        ],
        index=1,
    )
    max_tokens = st.slider("Max new tokens", 64, 512, 256)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7)
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful assistant. Answer questions clearly and concisely.",
    )


@st.cache_resource(show_spinner="Loading model... (this may take a minute)")
def load_pipeline(name: str):
    pipe = pipeline("text-generation", model=name, device_map="auto")
    # Remove default max_length so it doesn't conflict with max_new_tokens
    pipe.model.generation_config.max_length = None
    return pipe


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation for chat-style models
    chat_messages = [{"role": "system", "content": system_prompt}]
    chat_messages.extend(st.session_state.messages)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            pipe = load_pipeline(model_name)

            # Use GenerationConfig to avoid deprecation warning
            gen_config = GenerationConfig(
                max_new_tokens=max_tokens,
                max_length=None,
                temperature=temperature,
                do_sample=True,
            )

            if pipe.tokenizer.chat_template is not None:
                output = pipe(
                    chat_messages,
                    generation_config=gen_config,
                    return_full_text=False,
                )
                response = output[0]["generated_text"]
            else:
                text_prompt = ""
                for m in chat_messages:
                    role = m["role"].capitalize()
                    text_prompt += f"{role}: {m['content']}\n"
                text_prompt += "Assistant:"

                output = pipe(
                    text_prompt,
                    generation_config=gen_config,
                    return_full_text=False,
                )
                response = output[0]["generated_text"]

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

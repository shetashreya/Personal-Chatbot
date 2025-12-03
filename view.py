import streamlit as st
from model import read_file,chunk_text,embed_texts,build_faiss,retrieve,generate_answer,DATA_FILE

st.set_page_config(page_title="Shreya's Chat Bot", page_icon="📘")
st.markdown(
    """
    <h1 style="text-align: center;">Shreya's Chat Bot</h1>
    <p style="text-align: center; font-size:18px; color: gray;">
    Ask me anything about Shreya!
    </p>
    """,
    unsafe_allow_html=True
)

# Building the knowledge source base once
if "index" not in st.session_state:
    st.info("Loading Knowledge Source base...")
    text = read_file(DATA_FILE)
    st.session_state["chunks"] = chunk_text(text)
    embs = embed_texts(st.session_state["chunks"])
    st.session_state["index"] = build_faiss(embs)
    st.success("Knowledge Source base ready!")

# If Chat history is null
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi 👋 I'm AI assistant! Ask me anything you want to know about Shreya.",
        }
    ]

# Showing chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar=msg.get("avatar", None)):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Store + display user msg
    user_msg = {
        "role": "user",
        "content": prompt,
    }
    st.session_state["messages"].append(user_msg)
    with st.chat_message("user", avatar=user_msg.get("avatar", None)):
        st.markdown(prompt)

    # Generate + store assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            hits = retrieve(prompt, st.session_state["index"], st.session_state["chunks"], k=6)
            if not hits:
                response = "Apologies, I couldn’t find anything relevant."
            else:
                response = generate_answer(prompt, hits)
        st.markdown(response)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": response,
    })
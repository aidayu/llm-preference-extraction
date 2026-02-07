"""Streamlit chat application for preference elicitation.

嗜好把握のためのチャット対話システム。

元ファイル: preference_kg/experiments2/app.py
"""

import streamlit as st

from components.chat_session import ChatSession
from components.data_saver import save_dialogue_log


# Page config
st.set_page_config(
    page_title="嗜好把握対話システム",
    page_icon="💬",
    layout="centered",
)

# Custom CSS
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("⚙️ 設定")

    # User ID input
    user_id = st.text_input(
        "ユーザーID",
        value=st.session_state.get("user_id", ""),
        placeholder="例: user001",
    )

    if user_id:
        st.session_state.user_id = user_id

    st.divider()

    # Session controls
    if st.button("🔄 新しい会話を開始", use_container_width=True):
        if "chat_session" in st.session_state:
            # Save current session before starting new one
            session_data = st.session_state.chat_session.get_session_data()
            if session_data["dialogue_history"]:
                save_dialogue_log(session_data)
                st.success("会話を保存しました")

        # Reset session
        st.session_state.pop("chat_session", None)
        st.session_state.pop("messages", None)
        st.rerun()

    if st.button("💾 会話を保存して終了", use_container_width=True):
        if "chat_session" in st.session_state:
            session_data = st.session_state.chat_session.get_session_data()
            if session_data["dialogue_history"]:
                filepath = save_dialogue_log(session_data)
                st.success(f"保存しました: {filepath}")
            else:
                st.warning("保存する会話がありません")

    st.divider()
    st.caption("嗜好把握対話システム")

# Main chat area
st.title("💬 嗜好把握対話システム")

# Check if user ID is set
if not st.session_state.get("user_id"):
    st.info("👈 サイドバーでユーザーIDを入力してください")
    st.stop()

# Initialize chat session
if "chat_session" not in st.session_state:
    st.session_state.chat_session = ChatSession(user_id=st.session_state.user_id)
    st.session_state.messages = []

    # Add greeting message
    greeting = st.session_state.chat_session.get_greeting()
    st.session_state.messages.append({"role": "assistant", "content": greeting})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("メッセージを入力..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to messages and session
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_session.add_user_message(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            response = st.session_state.chat_session.generate_response()
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

import asyncio
import random
import streamlit as st
from dotenv import load_dotenv

from src.chain import ask_question, create_chain
from src.config import Config
from src.ingestor import IngestionPipeline
from src.model import create_llm
from src.retriever import create_retriever
from src.uploader import upload_files

load_dotenv()

@st.cache_resource(show_spinner=False)
def build_qa_chain(files):
    file_paths = upload_files(files)
    vector_store = IngestionPipeline().ingest(file_paths)
    llm = create_llm()
    retriever = create_retriever(llm, vector_store=vector_store)
    return create_chain(llm, retriever)


async def ask_chain(question: str, chain):
    full_response = ""
    assistant = st.chat_message(
        "assistant", avatar=str(Config.Path.IMAGES_DIR / "assistant-avatar.webp")
    )
    with assistant:
        message_placeholder = st.empty()
        documents = []
        async for event in ask_question(chain, question, session_id="session-id-42"):
            if type(event) is str:
                full_response += event
                message_placeholder.markdown(full_response)
            if type(event) is list:
                documents.extend(event)
        # for i, doc in enumerate(documents):
        #     with st.expander(f"Source #{i+1}"):
        #         st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def show_upload_documents():
    # Display Stratlytics logo
    st.image("images/logo.png", width=150)

    st.header("RagBase")
    st.subheader("Get answers from your documents")

    uploaded_files = st.file_uploader(
        label="Upload PDF files", type=["pdf"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.warning("Please upload PDF documents to continue!")
        st.stop()

    with st.spinner("Analyzing your document(s)..."):
        return build_qa_chain(uploaded_files)


def show_message_history():
    for message in st.session_state.messages:
        role = message["role"]
        avatar_path = (
            Config.Path.IMAGES_DIR / "assistant-avatar.webp"
            if role == "assistant"
            else Config.Path.IMAGES_DIR / "user-avatar.jpeg"
        )
        with st.chat_message(role, avatar=str(avatar_path)):
            st.markdown(message["content"])


def show_chat_input(chain):
    if prompt := st.chat_input("Ask your question here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message(
            "user",
            avatar=str(Config.Path.IMAGES_DIR / "user-avatar.jpeg")
        ):
            st.markdown(prompt)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ask_chain(prompt, chain))


# Page setup
st.set_page_config(page_title="RagBase", page_icon="")

# Apply custom CSS for button and logo styling
st.markdown(
    """
    <style>
        .st-emotion-cache-p4micv { width: 2.75rem; height: 2.75rem; }
        button { background-color: #007BFF; color: white; border: none; border-radius: 50%; }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize message history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! What do you want to know about your documents?",
        }
    ]

# Display upload interface, message history, and chat input
chain = show_upload_documents()
show_message_history()
show_chat_input(chain)

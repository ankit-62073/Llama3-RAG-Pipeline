import asyncio
import random
import streamlit as st
from dotenv import load_dotenv
import os 
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage

from src.chain import ask_question, create_chain
from src.config import Config
from src.ingestor import IngestionPipeline
from src.model import create_llm
from src.retriever import create_retriever
from src.uploader import upload_files
import time 

load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

def show_guide_document():
    """Function to display guide document content in a Streamlit modal"""
    with st.expander("📚 How to Use StratLytics Chatbot", expanded=True):
        st.markdown("""
        # StratLytics Chatbot User Guide

        ## Getting Started
        1. **Upload Documents** (Optional)
           - Click the file uploader in the sidebar
           - Select one or multiple PDF files
           - Wait for the processing confirmation

        ## Asking Questions
        - Type your question in the chat input at the bottom
        - If you've uploaded documents, the chatbot will answer based on their content
        - If no documents are uploaded, the chatbot will provide general responses
        
        ## Tips for Best Results
        - Ask clear, specific questions
        - For document-specific queries, ensure relevant PDFs are uploaded first
        - You can upload multiple documents at once
        
        ## Document Handling
        - Supported format: PDF files
        - Documents are processed securely
        - You can upload new documents at any time
        
        ## Need Help?
        If you encounter any issues:
        - Check if your documents are in PDF format
        - Ensure questions are clearly formulated
        - Try refreshing the page if uploads aren't working
        """)
        
        # Add a download button for a detailed PDF guide if needed
        try:
            # Read the local document file
            with open("images/Documentation.pdf", "rb") as file:
                # Create download button that uses the file data directly
                st.download_button(
                    label="📥 Download Detailed Guide",
                    data=file,
                    file_name="Documentation.pdf",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
        except FileNotFoundError:
            st.error("Guide document not found in the specified location. Please check if 'images/Documentation.docx' exists.")
        except Exception as e:
            st.error(f"Error accessing the guide document: {str(e)}")

@st.cache_resource(show_spinner=False)
def build_qa_chain(files):
    file_paths = upload_files(files)
    vector_store = IngestionPipeline().ingest(file_paths)
    llm = create_llm()
    retriever = create_retriever(llm, vector_store=vector_store)
    return create_chain(llm, retriever)

async def stream_response(text: str, message_placeholder):
    """Helper function to stream text word by word"""
    full_response = ""
    # Split text into words and add spaces back
    words = text.split(' ')
    for i, word in enumerate(words):
        full_response += word
        if i < len(words) - 1:  # Don't add space after last word
            full_response += " "
        message_placeholder.markdown(full_response + "▌")
        await asyncio.sleep(0.05)  # Adjust speed of typing
    return full_response

async def ask_chain(question: str, chain=None):
    full_response = ""
    assistant = st.chat_message(
        "assistant", avatar=str(Config.Path.IMAGES_DIR / "logo2.png")
    )
    with assistant:
        message_placeholder = st.empty()
        documents = []
        
        if chain:
            # Use RAG if chain exists (PDF uploaded)
            async for event in ask_question(chain, question, session_id="session-id-42"):
                if isinstance(event, str):
                    full_response += event
                    message_placeholder.markdown(full_response)
                if isinstance(event, list):
                    documents.extend(event)
        else:
            # Use direct LLM if no PDFs uploaded
            llm = create_llm()
            try:
                # Create a proper ChatOllama message
                messages = [HumanMessage(content=question)]
                response = await llm.agenerate([messages])
                # Stream the complete response
                full_response = await stream_response(
                    response.generations[0][0].text,
                    message_placeholder
                )
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                full_response = "I apologize, but I encountered an error processing your request."
                message_placeholder.markdown(full_response)

        # Remove the cursor after streaming is complete
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

def show_upload_documents():
    with st.sidebar:
        # Display Stratlytics logo
        st.image("images/logo.png", width=220)

        # st.header("RagBase")
        st.subheader("Get answers from your documents")

        uploaded_files = st.file_uploader(
            label="Upload PDF files (optional)", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Analyzing your document(s)..."):
                chain = build_qa_chain(uploaded_files)
                st.success(f"✅ {len(uploaded_files)} document(s) successfully processed! You can now ask questions about your documents.")
        
        # Add spacing
        st.markdown("---")
        
        # Replace the old guide button with new implementation
        if st.button("📚 Guide", use_container_width=True):
            show_guide_document()
        
        return chain if uploaded_files else None

def show_message_history():
    for message in st.session_state.messages:
        role = message["role"]
        avatar_path = (
            Config.Path.IMAGES_DIR / "logo2.png"
            if role == "assistant"
            else Config.Path.IMAGES_DIR / "user-avatar.jpeg"
        )
        with st.chat_message(role, avatar=str(avatar_path)):
            st.markdown(message["content"])

def run_async(coro):
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

def show_chat_input(chain):
    if prompt := st.chat_input("Ask any question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message(
            "user",
            avatar=str(Config.Path.IMAGES_DIR / "user-avatar.jpeg")
        ):
            st.markdown(prompt)
        run_async(ask_chain(prompt, chain))

# Page setup
st.set_page_config(
    page_title="StratLytics", 
    page_icon="images/logo2.png",
    layout="wide"
)

# Apply custom CSS for button and logo styling
st.markdown(
    """
    <style>
        /* Avatar container styling */
        .st-emotion-cache-p4micv {
            width: 4rem !important;
            height: 2.75rem !important;
            border-radius: 50%;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Avatar image styling */
        .st-emotion-cache-p4micv img {
            width: 100% !important;
            height: 100% !important;
            object-fit: cover !important;
            border-radius: 50%;
        }

        /* Guide button styling */
        .stButton button {
            background-color: #007BFF;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            width: 100%;
        }
        
        .stButton button:hover {
            background-color: #0056b3;
        }
        
        /* Other existing styles */
        button { 
            background-color: #007BFF; 
            color: white; 
            border: none; 
            border-radius: 50%; 
        }
        .main .block-container {
            padding-top: 2rem;
            max-width: 800px;
            margin: 0 auto;
        }

        /* Expander styling */
        .streamlit-expander {
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        .streamlit-expander .streamlit-expanderHeader {
            background-color: #f8f9fa;
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize message history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Stratlytics is ready to answer your query.",
        }
    ]

# Main content area title
st.title("StratLytics Chatbot")

# Display upload interface, message history, and chat input
chain = show_upload_documents()
show_message_history()
show_chat_input(chain)
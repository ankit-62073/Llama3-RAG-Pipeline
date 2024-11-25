import asyncio
import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_core.messages import HumanMessage

from src.auth import AuthManager
from src.chain import ask_question, create_chain
from src.config import Config
from src.ingestor import IngestionPipeline
from src.model import create_llm
from src.retriever import create_retriever
from src.uploader import upload_files

load_dotenv()

# Initialize authentication manager
auth_manager = AuthManager()

def show_guide_document():
    # with st.expander("📚 How to Use StratLytics Chatbot", expanded=True):
    st.markdown('''
    # StratLytics Chatbot User Guide
    
    ## Getting Started
    1. **Create an Account or Login**
        - New users: Click the "Sign Up" tab and create an account
        - Existing users: Use the "Login" tab with your credentials
    
    2. **Upload Documents** (Optional)
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
    - Each user has their own secure document storage
    
    ## Need Help?
    If you encounter any issues:
    - Check if your documents are in PDF format
    - Ensure questions are clearly formulated
    - Try refreshing the page if uploads aren't working
    ''')

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
def build_qa_chain(files, user_id):
    user_dir = Config.Path.VECTOR_STORES_DIR / f"user_{user_id}"
    file_paths = upload_files(files)
    vector_store = IngestionPipeline().ingest(file_paths, persist_directory=str(user_dir))
    llm = create_llm()
    retriever = create_retriever(llm, vector_store=vector_store)
    return create_chain(llm, retriever)

async def stream_response(text: str, message_placeholder):
    full_response = ""
    words = text.split(' ')
    for i, word in enumerate(words):
        full_response += word
        if i < len(words) - 1:
            full_response += " "
        message_placeholder.markdown(full_response + "▌")
        await asyncio.sleep(0.05)
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
            async for event in ask_question(chain, question, session_id=f"session-{st.session_state.user_id}"):
                if isinstance(event, str):
                    full_response += event
                    message_placeholder.markdown(full_response)
                if isinstance(event, list):
                    documents.extend(event)
        else:
            llm = create_llm()
            try:
                messages = [HumanMessage(content=question)]
                response = await llm.agenerate([messages])
                full_response = await stream_response(
                    response.generations[0][0].text,
                    message_placeholder
                )
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                full_response = "I apologize, but I encountered an error processing your request."
                message_placeholder.markdown(full_response)

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

def show_upload_documents():
    with st.sidebar:
        st.image(str(Config.Path.IMAGES_DIR / "logo.png"), width=220)
        st.subheader("Get answers from your documents")

        # Add radio button for upload type selection
        upload_type = st.radio(
            "Choose upload type:",
            ["Single File", "Multiple Files"],
            key="upload_type"
        )

        if upload_type == "Single File":
            uploaded_files = st.file_uploader(
                label="Upload a PDF file", 
                type=["pdf"], 
                accept_multiple_files=False
            )
            # Convert single file to list if it exists
            uploaded_files = [uploaded_files] if uploaded_files else []
        else:
            uploaded_files = st.file_uploader(
                label="Upload PDF files", 
                type=["pdf"], 
                accept_multiple_files=True
            )
        
        if uploaded_files and any(uploaded_files):  # Check if there are any valid files
            with st.spinner("Analyzing your document(s)..."):
                # Remove None values if any
                valid_files = [f for f in uploaded_files if f is not None]
                if valid_files:
                    chain = build_qa_chain(valid_files, st.session_state.user_id)
                    st.success(f"✅ {len(valid_files)} document(s) successfully processed!")
                else:
                    st.warning("No valid files were uploaded.")
                    return None
        
        st.markdown("---")
        
        return chain if uploaded_files and any(uploaded_files) else None

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

def show_auth_page():
    st.title("Welcome to StratLytics Chatbot")
    with st.sidebar:
        # if st.button("📚 Guide", use_container_width=True):
        show_guide_document()
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.header("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            user_id = auth_manager.authenticate_user(login_username, login_password)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.username = login_username
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": f"Hi {login_username}! Stratlytics is ready to answer your query.",
                    }
                ]
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        st.header("Sign Up")
        new_username = st.text_input("Username", key="new_username")
        new_password = st.text_input("Password", type="password", key="new_password")
        if st.button("Sign Up"):
            if auth_manager.register_user(new_username, new_password):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username already exists")

def main():
    st.set_page_config(
        page_title="StratLytics",
        page_icon=str(Config.Path.IMAGES_DIR / "logo2.png"),
        layout="wide"
    )
    
    # Hide Streamlit's default menu and footer
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Top right corner logout button */
            .logout-button {
                position: fixed;
                top: 0.5rem;
                right: 1rem;
                z-index: 999999;
            }
            
            .logout-button button {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 0.5rem 1rem;
                font-weight: 500;
            }
            
            .logout-button button:hover {
                background-color: #c82333;
            }
            
            .st-emotion-cache-p4micv {
                width: 4rem !important;
                height: 2.75rem !important;
                border-radius: 50%;
                overflow: hidden;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .st-emotion-cache-p4micv img {
                width: 100% !important;
                height: 100% !important;
                object-fit: cover !important;
                border-radius: 50%;
            }

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
            
            .main .block-container {
                padding-top: 2rem;
                max-width: 800px;
                margin: 0 auto;
            }

            .streamlit-expander {
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-bottom: 1rem;
            }
            
            .streamlit-expander .streamlit-expanderHeader {
                background-color: #f8f9fa;
                padding: 0.75rem 1rem;
                cursor: pointer;
            }
            
            .streamlit-expander .streamlit-expanderContent {
                padding: 1rem;
            }
            
            .element-container {
                margin-bottom: 1rem;
            }
            
            .stTextInput input {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 0.5rem;
            }
            
            .stTextInput input:focus {
                border-color: #007BFF;
                box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
            }
                
            /* Button Styles */
            .stButton > button {
                background-color: #007BFF !important;
                color: white !important;
                border: none !important;
                border-radius: 4px !important;
                padding: 0.5rem 1.2rem !important;
                font-weight: 500 !important;
                font-size: 1rem !important;
                line-height: 1.5 !important;
                text-align: center !important;
                white-space: nowrap !important;
                vertical-align: middle !important;
                cursor: pointer !important;
                user-select: none !important;
                transition: all 0.15s ease-in-out !important;
                width: auto !important;
                min-width: 100px !important;
                height: 38px !important;
                margin: 0 !important;
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
            
            .stButton > button:hover {
                background-color: #0056b3 !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
            }
        </style>
    """,
    unsafe_allow_html=True)

    if "user_id" not in st.session_state:
        show_auth_page()
    else:
        # Add logout button to top right corner
        st.markdown(
            '''
            <div class="logout-button">
                <form action="/" method="get" id="logout-form">
                    <button type="submit" onclick="document.cookie='logout=true; path=/'; localStorage.clear(); sessionStorage.clear();">
                        Logout
                    </button>
                </form>
            </div>
            ''',
            unsafe_allow_html=True
        )
        
        # Handle logout
        if st.session_state.get('logout_clicked', False):
            st.session_state.clear()
            st.rerun()
            
        chain = show_upload_documents()
        show_message_history()
        show_chat_input(chain)

# if __name__ == "__main__":
main()
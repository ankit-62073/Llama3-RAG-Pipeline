import re
from operator import itemgetter
from typing import List

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import Config
from src.session_history import get_session_history

from logger.logging import logging

# SYSTEM_PROMPT = """
# Utilize the provided contextual information to respond to the user question. If the answer is not found within the context, state that the answer cannot be found. Prioritize concise responsed (maximum of 3 sentences) and use a list where applicable. The contextual information is organized with the most relevant source appearing first. Each source is seperated by a horizontal rule (----).

# Context: {context}

# Use markdown formatting where appropriate.
# """

# SYSTEM_PROMPT = """
# Utilize the provided contextual information to respond to the user question. If the answer is not found within the context, explicitly state that the provided context is not mentioned in the documents. Prioritize concise responses (maximum of 3 sentences) and use a list where applicable. The contextual information is organized with the most relevant source appearing first. Each source is separated by a horizontal rule (----).

# Context: {context}

# Use markdown formatting where appropriate.
# """

# SYSTEM_PROMPT = """
# Utilize the provided contextual information to respond to the user question. If the answer is not found within the context, do not provide any response. Responses should only pertain to the information contained within the provided documents. Prioritize concise responses (maximum of 3 sentences) and use a list where applicable. The contextual information is organized with the most relevant source appearing first. Each source is separated by a horizontal rule (----).

# Context: {context}

# Use markdown formatting where appropriate.
# """

# SYSTEM_PROMPT = """
# Utilize the provided contextual information to respond to the user question. If the answer is not found within the context, do not provide any response. Responses should only pertain to the information contained within the provided documents. Prioritize concise responses (maximum of 3 sentences) and use a list where applicable. The contextual information is organized with the most relevant source appearing first. Each source is separated by a horizontal rule (----).

# Context: {context}

# Use markdown formatting where appropriate.
# """

SYSTEM_PROMPT = """
Respond strictly and exclusively based on the information contained within the uploaded document.

    Do not provide comparisons, inferences, or additional information not explicitly stated in the document.
    If the document does not address the query directly, respond with:
    "The document does not provide this information."

Response Guidelines:

    Respond in a concise manner (maximum of 3 sentences).
    Use only the language, phrasing, and terminology explicitly present in the document.
    Avoid introducing any external terms, concepts, or interpretations.
    When information is absent or incomplete, clearly state its absence as per the above directive.

Context: {context}

Use markdown formatting where appropriate."""

def remove_links(text: str) -> str:
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", text)


def format_documents(documents: List[Document]) -> str:
    texts = []
    for doc in documents:
        texts.append(doc.page_content)
        texts.append("----")

    return remove_links("\n".join(texts))


def create_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
    logging.info("Creating chain with LLM and retriever.")
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question")
                | retriever.with_config({"run_name": "context_retriever"})
                | format_documents
            )
            | prompt
            | llm
        )

        logging.info("Chain created successfully.")
        return RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        ).with_config({"run_name": "chain_answer"})
    except Exception as e:
        logging.error(f"Error creating chain: {e}")
        return None

async def ask_question(chain: Runnable, question: str, session_id: str):
    logging.info(f"Starting to ask question: {question}")
    try:
        async for event in chain.astream_events(
            {"question": question},
            config={
                "callbacks": [ConsoleCallbackHandler()] if Config.DEBUG else [],
                "configurable": {"session_id": session_id}
            },
            version="v2",
            include_names=["context_retriever", "chain_answer"],
        ):
            event_type = event["event"]
                
            if event_type == "on_retriever_start":
                logging.info("Fetching response from vectorstore.")

            if event_type == "on_retriever_end":
                logging.info("Retriever has finished.")
                yield event["data"]["output"]
                
            if event_type == "on_chain_stream":
                # logging.info("Streaming from chain.")
                yield event["data"]["chunk"].content
        logging.info("Completed asking question.")
    except Exception as e:
        logging.error(f"Error during ask_question: {e}")
        

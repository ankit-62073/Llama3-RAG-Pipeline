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

from src.ingestor import IngestionPipeline
from src.retriever import create_retriever
from src.model import create_llm

SYSTEM_PROMPT = """
Utilize the provided contextual information to respond to the user question. If the answer is not found within the context, state that the answer cannot be found. Prioritize concise responses (maximum of 3 sentences) and use a list where applicable. The contextual information is organized with the most relevant source appearing first. Each source is separated by a horizontal rule (---).

Context: {context}

Use markdown formatting where appropriate.
"""

def remove_links(text: str) -> str:
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", text)

def format_documents(documents: List[Document]) -> str:
    texts = [doc.page_content for doc in documents]
    return remove_links("\n----\n".join(texts))

def create_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
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

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    ).with_config({"run_name": "chain_answer"})

async def ask_question(chain: Runnable, question: str, session_id: str):
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
            if event_type == "on_retriever_end":
                yield event["data"]["output"]
            if event_type == "on_chain_stream":
                yield event["data"]["chunk"].content
    except Exception as e:
        print(f"Error during ask_question: {e}")


# def ask_question(chain: Runnable, question: str, session_id: str):
#     try:
#         output = []  # Collect outputs here
#         # Use the appropriate method for synchronous event handling
#         result = chain.invoke(
#             {"question": question},
#             config={
#                 "configurable": {"session_id": session_id}
#             }
#         )
        
#         # If the result is iterable, collect results
#         if isinstance(result, list):
#             output.extend(result)
#         else:
#             output.append(result)

#         return output  # Return the collected output
#     except Exception as e:
#         print(f"Error during ask_question: {e}")
#         return []  # Return an empty list on error



async def main():
    ingestion_obj = IngestionPipeline()
    ingestion_obj.ingest(["data2/data.pdf"])

    llm = create_llm()
    retriever = create_retriever(llm)

    question = "what is social control"
    session_id = "session-id-42"  # Replace with your actual session ID
    chain = create_chain(llm, retriever)

    # Collect the response from the async generator
    async for output in ask_question(chain, question, session_id):
        print(output)

    # response = ask_question(chain, question, session_id)
    
    # for output in response:
    #     print(output)  # Print each output


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

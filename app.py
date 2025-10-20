import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational Q&A with RAG")
api_key = st.text_input("Insert your API key",type="password")

if api_key:
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)

    session_id=st.text_input("Session ID",value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()


        context_q_system_prompt = ("Given a chat history and the latest user question"
                                   "Which may refer to the context in the chat history"
                                   "Provide a detailed standlone question that captures necessary context"
                                   "If you do not have enough information, simply give the original question.")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", context_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        contextualize_q_chain = (
            prompt | llm | StrOutputParser()
            
        )
        def reformulate_question(chat_history, user_question):
            return contextualize_q_chain.invoke({
                "chat_history": chat_history,
                "question": user_question
            })
        
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        system_prompt = ("You are a helpful AI assistant that helps people find information"
                         "from their documents. Use the following pieces of context to answer"
                         "the users question. If you don't know the answer, just say that you"
                         "don't know, don't try to make up an answer.")
        
        prompt1 = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context: {context}"),
            ("user", "Question: {question}"),
        ])

        qa_chain = (
            prompt1 | llm | StrOutputParser()   
        )
        def answer_question(retriever, session_id, user_question):
            # get session history
            session_history = get_session_history(session_id)
            chat_msgs = session_history.messages

            # reformulate question based on history
            standalone_question = reformulate_question(chat_msgs, user_question)

            # retrieve relevant docs
            relevant_docs = vectorstore.similarity_search(standalone_question, k=5)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # generate answer
            answer = qa_chain.invoke({
                "context": context,
                "question": standalone_question
            })

            # save messages to history
            session_history.add_user_message(user_question)
            session_history.add_ai_message(answer)

            return answer
        
        user_input = st.text_input("Ask a question about your documents:")
        if user_input:
            session_history = get_session_history(session_id)
            answer = answer_question(retriever, session_id, user_input)
            st.write(st.session_state.store)
            st.write("Assistant:", answer)
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")




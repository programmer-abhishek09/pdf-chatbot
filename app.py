import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import tempfile
from dotenv import load_dotenv

load_dotenv()

st.title("Chat with your PDF")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# File upload and indexing
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Check if this is a NEW file by comparing filename
    if st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("Reading and indexing your PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.read())
                tmp_path = f.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(pages)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
            )

            # Reset everything for the new PDF
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.chat_history = []        # clear old conversation
            st.session_state.current_file = uploaded_file.name  # remember filename

        st.success(f"Indexed: {uploaded_file.name} — chat history cleared!")
# Display chat history
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)
        if role == "assistant" and hasattr(message, "additional_kwargs"):
            sources = message.additional_kwargs.get("sources", "")
            if sources:
                st.caption(sources)
# Chat input
if st.session_state.vectorstore:
    question = st.chat_input("Ask something about your PDF...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)

        # Retrieve relevant chunks
        retriever = st.session_state.vectorstore.as_retriever()
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Build prompt with chat history
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer based on this context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        llm = ChatGroq(model_name="llama-3.1-8b-instant")
        chain = prompt | llm

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke({
                    "context": context,
                    "chat_history": st.session_state.chat_history,
                    "question": question
                })
                answer = response.content

            st.write(answer)

            # Show source pages
            pages_used = sorted(set(
                doc.metadata.get("page", 0) + 1
                for doc in relevant_docs
            ))
            source_text = ", ".join(f"p.{p}" for p in pages_used)
            st.caption(f"Sources: {source_text} — {st.session_state.current_file}")

        # Save to history
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(
            content=answer,
            additional_kwargs={"sources": f"Sources: {source_text} — {st.session_state.current_file}"}
        ))
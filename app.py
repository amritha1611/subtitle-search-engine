import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai import OpenAI

# Initialize embedding model
model_name = "sentence-transformers/all-mpnet-base-v2"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# Setting a connection with ChromaDB
db = Chroma(collection_name="vector_database",
              embedding_function=embedding_model,
              persist_directory="./chroma_db")

# Read OpenAI API Key
with open("C:/Users/Asus/Desktop/Code/Gen AI/Open AI/keys/.openai_api_key.txt") as f:
    OPENAI_API_KEY = f.read().strip()

client = OpenAI(api_key=OPENAI_API_KEY)

# Prompt template
system_prompt_template = """You are an AI assistant.  
You will receive a transcript of an audio recording in {user_input}.  
Match the transcript to the most relevant document from {retrieved_docs}.  
Return only the title and release year of the matching series or movie.  
If it is a series then return the episode and/or season no. also if available.  
Provide no additional text.  
"""

prompt_template = ChatPromptTemplate.from_template(system_prompt_template)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# Chat model
chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")

# Output parser
parser = StrOutputParser()

# Cached retrieval function
@st.cache_resource
def retrieve_documents(query: str):
    query_chunks = text_splitter.split_text(query)
    query_embedding = [embedding_model.embed_query(chunk) for chunk in query_chunks]
    return db.similarity_search_by_vector(query_embedding, k=3)

# Wrap retrieval logic in a lambda function
retrieve_docs = RunnableLambda(retrieve_documents)

# Define the RAG chain
rag_chain = {
    "retrieved_docs": retrieve_docs,
    "user_input": RunnablePassthrough()
} | prompt_template | chat_model | parser

# Streamlit UI
st.title("🎬 FlickMatch")
st.write("Upload an audio file, and the AI will match it to the most relevant movie/series.")

uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV)", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("Find Relevant Document"):
        with st.spinner("Transcribing audio..."):
            try:
                transcription_response = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe", 
                    file=uploaded_file,  # ✅ Pass file directly
                    response_format="text"
                )
                transcription_text = transcription_response.strip() if transcription_response else "No transcription available."
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                transcription_text = "Transcription failed."

        st.text_area("Transcription", transcription_text, height=150, key="transcription_display")

        if transcription_text and transcription_text != "Transcription failed.":
            with st.spinner("Searching for the best match..."):
                try:
                    retrieved_output = rag_chain.invoke(transcription_text)
                    st.success("Retrieved successfully!")
                    st.write("### Best Matched Movie/Series:")
                    st.write(retrieved_output)
                except Exception as e:
                    st.error(f"Error retrieving document: {e}")
        else:
            st.warning("Skipping retrieval as transcription failed.")
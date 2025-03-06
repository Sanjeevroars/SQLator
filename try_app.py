from dotenv import load_dotenv
import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile
from langchain.utilities import SQLDatabase 
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def init_database(user: str, password: str, host: str, name: str) -> SQLDatabase:
    db_uri = f"mysql+pymysql://{user}:{password}@{host}/{name}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template ="""
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else.
    
    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=api_key, temperature=0)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
  
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
    prompt = ChatPromptTemplate.from_template(template)

    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=api_key, temperature=0)    

    def clean_sql_string(sql_string):
        return sql_string.strip("```").lstrip("sql").strip()

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(clean_sql_string(vars["query"])),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

def play_tts(response_text):
    if response_text:
        tts = gTTS(response_text, lang="en")  
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name) 
        audio = AudioSegment.from_mp3(temp_audio.name) 
        play(audio)  

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I am your SQL Bot. Please connect your database to begin."),
    ]

load_dotenv()

st.set_page_config(page_title="SQL Chatbot", page_icon=":speech_balloon:")
st.title("SQL Chatbot")

with st.sidebar:
    st.subheader("Settings")
    st.subheader("Connect your database to start chatting.")

    st.text_input("Host", value="localhost", key="host")
    st.text_input("User", value="root", key="user")
    st.text_input("Password", type="password", value="root", key="password")
    st.text_input("Database", value="atliq_tshirts", key="name")

    if st.button("Connect"):
        with st.spinner("Connecting to the Database..."):
            db = init_database(
                st.session_state["user"],
                st.session_state["password"],
                st.session_state["host"],
                st.session_state["name"]
            )
            st.session_state.db = db
            st.success("Connected to the database!")

for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.markdown(message.content)

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Please speak your query.")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError:
            return "Error connecting to Google Speech Recognition."

if st.button("🎤 Speak Now"):
    speech_text = recognize_speech()
    if speech_text:
        st.session_state.chat_history.append(HumanMessage(content=speech_text))

        with st.chat_message("Human"):
            st.markdown(speech_text)

        with st.chat_message("AI"):
            response = get_response(speech_text, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)
            play_tts(response)  # Play AI response as speech

        st.session_state.chat_history.append(AIMessage(content=response))

user_query = st.chat_input("Type a message...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        play_tts(response)  # Play AI response as speech

    st.session_state.chat_history.append(AIMessage(content=response))

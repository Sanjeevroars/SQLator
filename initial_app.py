from dotenv import load_dotenv
import os
from langchain.utilities import SQLDatabase 
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

def init_database(user: str, password: str, host: str, name: str)-> SQLDatabase:
    db_uri = f"mysql+pymysql://{user}:{password}@{host}/{name}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template ="""
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    api_key = os.getenv("GOOGLE_API_KEY")
    llm  = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash', api_key=api_key, temperature=0)

    def get_schema(_):
        schema = db.get_table_info()
        return schema
    
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
    llm  = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash', api_key=api_key, temperature=0)    
  
    def clean_sql_string(sql_string):
        cleaned_string = sql_string.strip("```").lstrip("sql").strip()
        return cleaned_string

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
  


if "chat_history" not in st.session_state:
    st.session_state.chat_history =[
        AIMessage(content = "Hello I am your SQL Bot! Please first connnect your database."),
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
    if isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message>...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))
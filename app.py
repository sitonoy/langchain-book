import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder


load_dotenv()

def create_agent_chain():
    chat = ChatOpenAI(
        model_name = os.environ["OPENAI_API_MODEL"],
        temperature = os.environ["OPENAI_API_TEMPERATURE"],
        streaming = True,
        
    )
    
    agent_kwargs = {
        "extra_prompt_messages":[MessagesPlaceholder(variable_name="memory")],
    }
    
    memory = ConversationBufferMemory(memory_key = "memory", return_messages = True)
    
    tools = load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(
        tools,
        chat,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs = agent_kwargs,
        memory = memory,
    )
    
#一度だけAgentを初期化することで空のMemoryを作成する
if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

#Streamlitのタイトル
st.title("langchain-streamlit-app")

#入力プロンプト設置及び入力欄に表示する文言の定義
prompt = st.chat_input("What's up?")

#会話履歴の表示
#session_stateが空の場合は空のまま
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
#promptに入力が合った場合はsession_state.messagesに追加していく
if prompt:
    st.session_state.messages.append({"role":"user","content": prompt})
    #ユーザのインプットとしてマークダウンとしてUIに表示
    with st.chat_message("user"):
        st.markdown(prompt)
    
    #LLM側の対応をマークダウンとしてUIに表示しsession_state.messagesに追加
    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        agent_chain = create_agent_chain()
        response = st.session_state.agent_chain.run(prompt,callbacks=[callback])
        st.markdown(response)
    st.session_state.messages.append({"role":"assinstant","content": response})

import time

import boto3
import streamlit as st
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory

# Create a session using the default profile
session = boto3.session.Session()

# Get the region
REGION = session.region_name

st.title(f"ChatBedrock: Region={REGION}")

# Setup bedrock runtime
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)

# Define bedrock (this is only needed for listing models)
bedrock = boto3.client(
    service_name="bedrock",
    region_name=REGION,
)

def list_models():
    """
    Purpose:
        List all the models available in Bedrock
    Args/Requests:
         None
    Return:
        None
    """
    response = bedrock.list_foundation_models(byInferenceType='PROVISIONED')
    for model in response['modelSummaries']:
        print(model['modelId'])

    return

@st.cache_resource
def load_llm():
    
    try:
       llm = Bedrock(client=bedrock_runtime, model_id="anthropic.claude-v2")
       llm.model_kwargs = {"temperature": 0.7, "max_tokens_to_sample": 2048}
    except Exception as e:
        print(e)
        exit(1)

    model = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

    return model

list_models()

model = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # prompt = prompt_fixer(prompt)
        try:
            result = model.predict(input=prompt)
        except Exception as e:
            print(f"{e}: ModelId={model.llm.model_id}")
            exit(1)

        # Simulate stream of response with milliseconds delay
        for chunk in result.split(' '): # fix for https://github.com/streamlit/streamlit/issues/868
            full_response += chunk + ' '
            if chunk.endswith('\n'):
                full_response += ' '
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

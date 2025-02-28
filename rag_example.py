import json

import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

# Create a session using the default profile
session = boto3.session.Session()

# Get the region
REGION = session.region_name

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)

sentences = [
    # Pets
    "Your dog is so cute.",
    "How cute your dog is!",
    "You have such a cute dog!",
    # Cities in the US
    "New York City is the place where I work.",
    "I work in New York City.",
    # Color
    "What color do you like the most?",
    "What is your favourite color?",
]


def claude_prompt_format(prompt: str) -> str:
    # Add headers to start and end of prompt
    return "\n\nHuman: " + prompt + "\n\nAssistant:"

# Call Claude model
def call_claude(prompt):
    prompt_config = {
        "prompt": claude_prompt_format(prompt),
        "max_tokens_to_sample": 4096,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 0.5,
        "stop_sequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-v2"
    accept = "application/json"
    contentType = "application/json"

    print("Before Claude")
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    print("After Claude")
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completion")
    return results

MODEL_ID="amazon.titan-embed-text-v1"
def rag_setup(query):
    print("Before TITAN")
    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id=MODEL_ID,
    )
    print("After TITAN")
    try:
        local_vector_store = FAISS.from_texts(sentences, embeddings)
    except Exception as e:
        print(f"{e} ModelId={MODEL_ID}")
        return "Error"
    
    docs = local_vector_store.similarity_search(query)
    context = ""

    for doc in docs:
        context += doc.page_content

    prompt = f"""Use the following pieces of context to answer the question at the end.

    {context}

    Question: {query}
    Answer:"""

    print("About to call Claude")
    return call_claude(prompt)


query = "What type of pet do I have?"
print(query)
print(rag_setup(query))

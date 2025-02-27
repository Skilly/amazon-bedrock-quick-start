import streamlit as st
import boto3
import json
import base64
import io
from PIL import Image


REGION = "us-east-1"

st.title("Building with Bedrock")  # Title of the application
st.subheader(f"Image Generation Demo (Region={REGION})")

# List of Stable Diffusion Preset Styles
sd_presets = [
    "None",
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]

# Define bedrock
bedrock = boto3.client(
    service_name="bedrock",
    region_name=REGION,
)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
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


def call_claude_sonet(base64_string):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_string,
                        },
                    },
                    {"type": "text", "text": "Provide a caption for this image"},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)
                        
    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    print("Before Claude")
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    print("After Claude")
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results


# Bedrock api call to stable diffusion
def generate_image_sd(text, style):
    """
    Purpose:
        Uses Bedrock API to generate an Image
    Args/Requests:
         text: Prompt
         style: style for image
    Return:
        image: base64 string of image
    """
    body = {
        "text_prompts": [{"text": text}],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 50,
        "style_preset": style,
    }

    if style == "None":
        del body["style_preset"]

    body = json.dumps(body)


    modelId = "stability.stable-diffusion-xl-v1"
    accept = "application/json"
    contentType = "application/json"

    print("Before  Diffusion")
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    print("After  Diffusion")
    response_body = json.loads(response.get("body").read())

    results = response_body.get("artifacts")[0].get("base64")
    return results


def convert_base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return img


def generate_image_titan(text):
    """
    Purpose:
        Uses Bedrock API to generate an Image using Titan
    Args/Requests:
         text: Prompt
    Return:
        image: base64 string of image
    """
    body = {
        "textToImageParams": {"text": text},
        "taskType": "TEXT_IMAGE",
        "imageGenerationConfig": {
            "cfgScale": 10,
            "seed": 0,
            "quality": "standard",
            "width": 512,
            "height": 512,
            "numberOfImages": 1,
        },
    }

    body = json.dumps(body)
    modelId = "amazon.titan-image-generator-v1"
    accept = "application/json"
    contentType = "application/json"

    print("Before  Titan")
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    print("After  Titan")
    response_body = json.loads(response.get("body").read())

    results = response_body.get("images")[0]
    return results


model = st.selectbox("Select model", ["Stable Diffusion", "Amazon Titan"])

list_models()


if model == "Stable Diffusion":

    style = st.selectbox("Select style", sd_presets)
    prompt = st.text_input("Enter prompt")

    if st.button("Generate"):

        results = generate_image_sd(prompt, style)

        # use claude to describe image
        desc_image = call_claude_sonet(results)

        img = convert_base64_to_image(results)
        st.image(img, caption=desc_image)


elif model == "Amazon Titan":
    prompt = st.text_input("Enter prompt")

    if st.button("Generate"):
        results = generate_image_titan(prompt)
        # use claude to describe image
        desc_image = call_claude_sonet(results)

        img = convert_base64_to_image(results)
        st.image(img, caption=desc_image)

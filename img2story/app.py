from dotenv import find_dotenv, load_dotenv
from transformers import pipeline, set_seed
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#img2text
def img2txt (url):
    
    img2txt = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text = img2txt(url, max_new_tokens=50)[0]["generated_text"]
    
    #print(text)
    return text

#llm
def generate_story(text):
    
    # Load the model
    gpt2_pipeline = pipeline("text-generation", model="openai-community/gpt2")
    
    # Set seed for reproducibility
    set_seed(40)
    
    # Generate text
    generated_text = gpt2_pipeline(text, max_length=200, temperature=0.7, truncation=True)
    generated_text = generated_text[0]['generated_text']
    # Find the index of the first occurrence of '\n' character
    newline_index = generated_text.find('\n')
    
    # Extract the text up to the newline character
    if newline_index != -1:
        generated_text = generated_text[:newline_index]
    
    #print('text generated: ',generated_text) 
    return generated_text   
    

#text2speech
def text2speech (story):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payload = {
        "inputs": story
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    with open ('audio.flac', 'wb') as file:
        file.write(response.content)

text = img2txt("1_DyE61-rl-Bz0b4ONTWIioA.jpg")
story = generate_story(text)
text2speech(story)

#UI
def main():
    st.set_page_config(page_title='img 2 audio story', page_icon="🤡")
    
    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file.name, caption="Uploaded Image", use_column_width=True)
        scenario = img2txt(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)
        
        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)
        
        st.audio("audio.flac")
        
if __name__ == '__main__':
    main()
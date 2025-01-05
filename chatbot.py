import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import replicate
import os

# Set the Replicate API token
os.environ["REPLICATE_API_TOKEN"] = "r8_CyGY6JSgDk8CZyKpIXH6tV6pVvaY2E82RcIYZ"

# Load BLIP Processor and Model for Image Captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define LLaMA parameters
pre_prompt = ("You are a helpful assistant specialized in describing images and answering questions "
              "based on their content. Respond conversationally and in detail.")

# Streamlit app setup
st.title("Conversational Image Recognition Chatbot")
st.write("Upload an image and ask questions about it. The chatbot will generate a description and "
         "respond to your queries!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Generate image caption using BLIP
    st.write("Generating image description...")
    inputs = blip_processor(img, return_tensors="pt")
    blip_output = blip_model.generate(**inputs)
    image_caption = blip_processor.decode(blip_output[0], skip_special_tokens=True)

    # Display the generated caption
    st.subheader("Image Description:")
    st.write(image_caption)

    # User input for conversational questions
    user_question = st.text_input("Ask a question about the image:")

    if user_question:
        # Generate conversational response using LLaMA
        st.write("Generating response...")

        # Combine the pre-prompt, image description, and user question
        llama_prompt = (f"{pre_prompt} Here is a description of the image: '{image_caption}'. "
                        f"Now, answer the following question based on this description:\n\n"
                        f"User: {user_question}\n\nAssistant:")

        # Call the LLaMA model through Replicate
        try:
            llama_output = replicate.run(
                "meta/llama-2-13b-chat",
                input={
                    "prompt": llama_prompt,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_length": 150,
                    "repetition_penalty": 1.0,
                }
            )

            # Concatenate and display the response
            response = "".join(llama_output)
            st.subheader("Chatbot Response:")
            st.write(response)

        except Exception as e:
            st.error("Error generating response from LLaMA model.")
            st.error(str(e))

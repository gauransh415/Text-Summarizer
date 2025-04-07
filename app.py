import streamlit as st
from transformers import pipeline
import torch
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Handle environment variables to prevent issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    st.set_page_config(page_title="Text Summarizer", layout="wide")
    
    st.title("Text Summarizer")
    st.write("Enter your text below to get a summarized version using BART large CNN model")
    
    # Create a text area for input
    text_input = st.text_area("Input Text", height=300, placeholder="Paste your text here")
    
    # Sliders for controlling summarization parameters
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Maximum summary length", 30, 500, 150)
    with col2:
        min_length = st.slider("Minimum summary length", 10, 100, 40)
    
    # Summarize button
    if st.button("Summarize"):
        if text_input:
            with st.spinner("Generating summary... This may take a moment to download the model on first run."):
                try:
                    # Initialize the device to use CPU for better compatibility
                    device = -1  # CPU
                    
                    # Create the summarizer with PyTorch backend
                    summarizer = pipeline(
                        "summarization", 
                        model="facebook/bart-large-cnn", 
                        framework="pt",
                        device=device
                    )
                    
                    # Generate summary
                    summary = summarizer(text_input, max_length=max_length, min_length=min_length)
                    
                    # Display the summary
                    st.subheader("Summary")
                    st.write(summary[0]['summary_text'])
                    
                    # Display statistics
                    st.subheader("Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"Original text length: {len(text_input)} characters")
                    with col2:
                        st.info(f"Summary length: {len(summary[0]['summary_text'])} characters")
                    with col3:
                        compression = round((1 - len(summary[0]['summary_text']) / len(text_input)) * 100)
                        st.info(f"Compression: {compression}%")
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error("Please make sure you have installed the correct versions of the required packages.")
                    st.code("pip install streamlit==1.32.0 torch==2.2.0 transformers==4.38.2")
        else:
            st.warning("Please enter some text to summarize")

    # Example text section
    with st.expander("Example Text"):
        example_text = """I measure the effectiveness of searches by how long it takes me to find what I was actually looking for. By that measure, Google has been steadily getting worse.

When I search for something on Kagi, the correct result is in the first 2 links 95% of the time. It's in the top 5 links 99% of the time. That just doesn't happen with Google, Bing, etc.

The consistently great results page is further boosted by the search personalization I control. I've told Kagi that any results from Stack Overflow or Medium should be weighted higher, and blocked other sites I don't care to see results from. No ads. Objectively better search results. Of all the subscriptions I pay for, this is the hill I will die on."""
        
        if st.button("Use Example Text"):
            # Use a different key to avoid conflict with the other button
            st.session_state["example_text"] = example_text
            st.experimental_rerun()
    
    # Set the example text in the text area if it exists in session state
    if "example_text" in st.session_state:
        # Since we can't directly set the value, we need to use session state
        st.session_state["text_input"] = st.session_state["example_text"]

    # Add information about the model
    with st.expander("About this app"):
        st.write("""
        This app uses the BART-large-CNN model from Facebook for text summarization.
        The model is designed to generate concise summaries while preserving the key information from the original text.
        
        **Technical Details:**
        - Model: facebook/bart-large-cnn
        - Backend: PyTorch (CPU mode for compatibility)
        - The model was trained on CNN Daily Mail dataset, which consists of news articles and their summaries.
        """)

if __name__ == "__main__":
    main()
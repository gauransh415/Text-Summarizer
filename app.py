import streamlit as st
from transformers import pipeline

def main():
    st.set_page_config(page_title="Text Summarizer", layout="wide")
    
    st.title("Text Summarizer")
    st.write("Enter your text below to get a summarized version using BART large CNN model")
    
    # Create a text area for input
    text_input = st.text_area("Input Text", height=300, 
                             placeholder="Paste your text here...")
    
    # Sliders for controlling summarization parameters
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Maximum summary length", 30, 500, 150)
    with col2:
        min_length = st.slider("Minimum summary length", 10, 100, 40)
    
    # Summarize button
    if st.button("Summarize"):
        if text_input:
            with st.spinner("Generating summary..."):
                try:
                    # Load the summarization model
                    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                    
                    # Generate summary
                    summary = summarizer(text_input, max_length=max_length, min_length=min_length)
                    
                    # Display the summary
                    st.subheader("Summary")
                    st.write(summary[0]['summary_text'])
                    
                    # Display statistics
                    st.subheader("Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"Original text length: {len(text_input)} characters")
                    with col2:
                        st.info(f"Summary length: {len(summary[0]['summary_text'])} characters")
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to summarize")

    # Add information about the model
    with st.expander("About this app"):
        st.write("""
        This app uses the BART-large-CNN model from Facebook for text summarization.
        The model is designed to generate concise summaries while preserving the key information from the original text.
        """)

if __name__ == "__main__":
    main()
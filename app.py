import streamlit as st
import requests
import uuid
import base64
import gc

# Reset chat session
def reset_chat():
    """Reset the chat session state."""
    st.session_state.messages = []
    st.session_state.uploaded_file_name = None
    st.session_state.pdf_display = None
    st.session_state.pipeline_ready = False
    gc.collect()

# Initialize unique session ID if not present
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()

# Initialize chat session state if not present
if "messages" not in st.session_state:
    reset_chat()

# Backend API URL
BASE_URL = "http://127.0.0.1:8000"  # Replace with your FastAPI backend URL

def display_pdf(file):
    """Display a PDF file within the Streamlit app."""
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>"""
    st.session_state.pdf_display = pdf_display

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Your Documents")

    # File uploader
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        # Check if a new file is uploaded
        if st.session_state.uploaded_file_name != uploaded_file.name:
            try:
                # Send file to backend for processing
                response = requests.post(
                    f"{BASE_URL}/upload", files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                )

                if response.status_code == 200:
                    # Successfully uploaded and processed
                    st.success(f"{uploaded_file.name} uploaded and processed successfully!")
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.pipeline_ready = True
                    display_pdf(uploaded_file)
                else:
                    # Error during processing
                    error_message = response.json().get("error", "Unknown error occurred during processing.")
                    st.error(f"Error: {error_message}")
                    st.session_state.pipeline_ready = False

            except Exception as e:
                # Handle exceptions
                st.error(f"An error occurred: {e}")
                st.session_state.pipeline_ready = False

    # Display PDF preview if available
    if st.session_state.pdf_display:
        st.markdown(st.session_state.pdf_display, unsafe_allow_html=True)

# Main app layout
col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with {st.session_state.uploaded_file_name or 'Docs'} using DeepSeek-R1")

with col2:
    st.button("Clear ↺", on_click=reset_chat)  # Button to clear chat

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input for chat
if prompt := st.chat_input("Ask a question about the document:"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response placeholder
    with st.chat_message("assistant"):
        # Stream response from backend
        if st.session_state.pipeline_ready and st.session_state.uploaded_file_name:
            try:
                # Construct the full path of the uploaded PDF file
                pdf_path = st.session_state.uploaded_file_name

                # Send question to backend for answer generation
                response = requests.post(
                    f"{BASE_URL}/ask",
                    json={"question": prompt, "pdf_name": pdf_path},  # Send correct JSON structure
                    stream=True
                )

                if response.status_code == 200:
                    def response_generator():
                        for chunk in response.iter_content(chunk_size=None):
                            yield chunk.decode("utf-8")

                    full_response = st.write_stream(response_generator())
                else:
                    # Error during response generation
                    error_message = response.json().get("error", "Unknown error")
                    full_response = f"Error: {error_message}"
                    st.markdown(full_response)
            except Exception as e:
                # Handle exceptions
                full_response = f"An error occurred: {e}"
                st.markdown(full_response)
        else:
            # Handle case where pipeline is not ready
            full_response = "No PDF file uploaded or pipeline not ready. Please upload a PDF to proceed."
            st.markdown(full_response)

    # Save assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

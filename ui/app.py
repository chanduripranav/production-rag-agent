import streamlit as st
import requests

# Constants
API_URL = "http://127.0.0.1:8000/query"

# Title of the application
st.title("Ask My Docs — Agentic RAG")
st.markdown("Welcome! Type your question below to search across the indexed documents.")

# Form for user input
with st.form(key="query_form"):
    user_question = st.text_input("What would you like to know?")
    submit_button = st.form_submit_button(label="Ask")

# Execute when the user clicks 'Ask'
if submit_button:
    if not user_question.strip():
        st.warning("Please enter a question to ask.")
    else:
        # Show a spinner while waiting for the response
        with st.spinner("Analyzing documents and generating an answer..."):
            try:
                # Send the POST request to the FastAPI backend
                response = requests.post(
                    API_URL, 
                    json={"question": user_question}, 
                    timeout=120
                )
                
                # Check for successful response
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display the generated Answer
                    st.subheader("Answer")
                    st.write(data.get("answer", "No answer provided."))
                    
                    # Display Citation Validation Status
                    is_valid = data.get("citation_validation_passed", False)
                    if is_valid:
                        st.success("✅ Citation Validation Passed: The claims are supported by the documents.")
                    else:
                        st.error("❌ Citation Validation Failed: Some claims may not be fully supported by the documents.")
                        
                    # Display Citations
                    citations = data.get("citations", [])
                    if citations:
                        st.subheader("Citations")
                        for idx, citation in enumerate(citations, start=1):
                            with st.expander(f"[{idx}] {citation.get('source_file')} (Page {citation.get('page_number')})"):
                                st.write("**Text Segment:**")
                                st.write(f"_{citation.get('text_preview')}_")
                    else:
                        st.info("No citations were provided for this answer.")
                        
                else:
                    st.error(f"Backend Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                # Handle connection errors (e.g. FastAPI isn't running)
                st.error("Could not connect to the backend. Please ensure the FastAPI server is running on http://127.0.0.1:8000.")
                st.exception(e)

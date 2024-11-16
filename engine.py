import streamlit as st
from google import generativeai as genai
import time

st.title("AI Tutor")

st.sidebar.header("Google Gemini API Key")
api_key = st.sidebar.text_input(type="password", placeholder="Enter your LLM API key", label='api_key', label_visibility="hidden")

st.sidebar.header("Student Info")
school_selected = st.sidebar.text_input("Enter School Name")
class_selected = st.sidebar.selectbox("Select Class", ["Class 9", "Class 10", "Class 11", "Class 12"])
syllabus_selected = st.sidebar.selectbox("Select Syllabus", ["CBSE", "ICSE", "State Board"])
subject_selected = st.sidebar.multiselect("Choose Subjects", ["Maths", "Physics", "Chemistry", "Biology", "English"])

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash-8b', generation_config=genai.types.GenerationConfig(
    temperature=0.1,
    candidate_count=1,
    stop_sequences=None,
    max_output_tokens=1000,
    response_mime_type='text/plain')
)
chat = model.start_chat(history=[])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)
        
if api_key :
    if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = chat.send_message(prompt)
        with st.chat_message("assistant") :
            st.write_stream(stream_data(response.text))
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        
else:
    api_key = st.chat_input("Input API Key", )  
    row1 = st.columns(3)
    row1[0].container(height=100).html("<h5>Previous year Question paper for physics</h5>")
    row1[1].container(height=100).html("<h5>Newtons 2nd law of motion</h5>")
    row1[2].container(height=100).html('<h5>Periodic table</h5>')
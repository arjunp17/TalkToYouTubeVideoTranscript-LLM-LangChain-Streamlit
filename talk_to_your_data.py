import streamlit as st
import os
import openai
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
#from IPython.display import display, Markdown
from langchain.document_loaders import YoutubeLoader


def generate_response(file_url, openai_api_key, query_text):
    # Load document if file is uploaded
    if file_url is not None:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        openai.api_key  = os.getenv('OPENAI_API_KEY')
        loader = YoutubeLoader.from_youtube_url(file_url, add_video_info=False)
        index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])
        return index.query(query)





# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
file_url = st.text_input('Enter your YouTube video URL:', placeholder = 'YouTube video URL.')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not file_url)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (file_url and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(file_url and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(file_url, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)

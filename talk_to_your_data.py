import streamlit as st
import os
import openai
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
#from langchain.vectorstores import DocArrayInMemorySearch
#from langchain.indexes import VectorstoreIndexCreator
#from IPython.display import display, Markdown
from langchain.document_loaders import YoutubeLoader


def generate_response(file_url, openai_api_key, query_text):
    if file_url is not None:
        # Set OpenAI API key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        openai.api_key  = os.getenv('OPENAI_API_KEY')
        # Select LLM Model
        llm = ChatOpenAI(temperature = 0.0)
        # Load YouTube video transcript from the given url
        loader = YoutubeLoader.from_youtube_url(file_url, add_video_info=False)
        docs = loader.load()
        # Split documents into chunks
        #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        #texts = text_splitter.create_documents(docs)
        # Select embeddings
        embeddings = OpenAIEmbeddings()
        # Create a vectorstore from documents
        db = Chroma.from_documents(docs, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa_stuff = RetrievalQA.from_chain_type(llm=llm, 
                                                chain_type="stuff", 
                                                retriever=retriever, 
                                                verbose=False)
        return qa_stuff.run(query_text)





# Page title
st.set_page_config(page_title='🦜🔗 Talk to your YouTube Video')
st.title('🦜🔗 Talk to your YouTube Video')

# URL Text
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

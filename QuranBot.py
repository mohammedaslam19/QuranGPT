import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
st.header(" بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ")
pdf = "quraan_english.pdf"

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://w.forfun.com/fetch/f5/f5934c4cb24d5fb3e44a14b735ba6b75.jpeg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)


if pdf is not None:
    pdf_object = PdfReader(pdf)
    text = ""
    for page in pdf_object.pages[:50]:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text=text)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embedding)

    query = st.text_input("", placeholder="How are you feeling today?")

    template = f'''take in the user's query and according to the overall emotion and sentiment of the query
         provide a verse or a couple of verses if continuity is required fully written out without your interpretation
         to provide comfort or relatability to the user provide chapter and verse number at the end this is the query 
         {query}'''

    prompt = PromptTemplate(input_variables={query}, template=template)

    if query:
        similar_chunks = vectorstore.similarity_search(query=query, k=2)
        llm = OpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        generated_prompt = prompt.format()

        response = chain.run(input_documents=similar_chunks, question=generated_prompt)

        st.write(response)
        st.write(" Ref docs :")
        st.write(similar_chunks[0])
        st.write(similar_chunks[1])

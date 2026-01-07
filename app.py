import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

# Configuração da Página
st.set_page_config(page_title="Personal AI Trainer", page_icon="🏋️‍♂️")
st.title("🏋️‍♂️ Seu Personal Trainer IA")
st.markdown("Suba seu documento de treino/dieta e tire suas dúvidas.")

# 1. Configuração da API (Grátis no Google AI Studio)
api_key = st.sidebar.text_input("Cole sua Google API Key aqui:", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Upload do Arquivo
    arquivo_subido = st.file_uploader("Escolha seu arquivo de treino (.docx)", type=['docx'])

    if arquivo_subido:
        # Salvar arquivo temporariamente para o Loader ler
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(arquivo_subido.getvalue())
            caminho_tmp = tmp.name

        # Processamento (RAG)
        with st.spinner("Analisando seu plano..."):
            loader = Docx2txtLoader(caminho_tmp)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever()

            # Configurar a IA
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            template = "Use o contexto para responder: {context}\nPergunta: {question}"
            prompt = ChatPromptTemplate.from_template(template)
            
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )

        st.success("Tudo pronto! Pode perguntar.")
        
        # Chat
        pergunta = st.text_input("O que você quer saber sobre seu treino/dieta?")
        if pergunta:
            resposta = chain.invoke(pergunta)
            st.write("### Resposta:")
            st.info(resposta)
else:
    st.warning("Insira a chave de API na lateral para começar.")
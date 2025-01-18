from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Langchain setup
EMBEDDING_MODEL = "llama3.2"
LLM_MODEL = "llama3.2"

def create_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore

vectorstore = create_vectorstore("FastAPI/emsi.pdf")

def setup_chain(vectorstore):
    llm = OllamaLLM(model=LLM_MODEL)
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

chain = setup_chain(vectorstore)

# FastAPI App
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def process_query(query_request: QueryRequest):
    try:
        query = query_request.query
        response = chain.invoke(query)
        return {"response": response["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # use 127.0.1.0

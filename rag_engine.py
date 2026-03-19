import os
from dotenv import load_dotenv
from langchain_community.documentloaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbiddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplete

load_dotenv()

INDEX_PATH = 'faiss_index'

def build_index(pdf_path: str):

    # Build a vector index from a pdf 
    """Load PDF → split → embed → save FAISS index to disk."""

    # Load PDF pages
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # split into 500 character chunks with 50-char overlaps
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, chunk_overlap = 50
    )
    chunks= splitter.split_document(docs)

    # Embed and store
    embeddings = OpenAIEmbiddings(model = 'text-embedding-3-small')
    vectorstore = FAISS.from_documents(chunks,embeddings)
    vectorstore.save_local(INDEX_PATH)

    return len(chunks) # return chunk count for UI feedback

# Load Saved index

def load_index():
    """Load existing FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        INDEX_PATH, embeddings,
        allow_dangerous_deserialization=True
    )


# Build the RAG Chain 

def build_rag_chain(vectorstore):
     """Return a runnable RAG chain given a loaded vectorstore."""
    retriver = vectorstore.as_retriver(search_keywargs = {'k':'3'})

    # Format docs — include source filename for citation
    def doc_format(docs):
        parts = []
        for doc in docs:
            src = doc.metadeta.get('source','unknown')
            parts.append(f'[Source:{src}]\n{doc.page_content}')
        return \n\n.join(parts)

    prompt = ChatPromptTemplete.from_messages([
        ("system", """You are a helpful assistant that answers questions
based ONLY on the provided document context.
If the answer is not in the context, say:
'I could not find that information in the uploaded document.'
Always mention which page or source the answer came from.

Context:
{context}"""),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model='gpt-4o-mini', Temperature=0)

    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── Step 4: Ask a question ──────────────────────────────────────
def ask(chain, question: str) -> str:
    """Invoke the chain with a question, return answer string."""
    return chain.invoke(question)
     



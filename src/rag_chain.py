from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.config import LLM_MODEL

def get_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1)
    
    system_prompt = (
        "You are an AI Customer Support Specialist. Use the following context to answer accurately.\n"
        "If you don't know the answer, say you don't know and provide support@company.com.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    return create_retrieval_chain(retriever, question_answer_chain)
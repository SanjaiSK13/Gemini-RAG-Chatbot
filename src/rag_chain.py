import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.config import LLM_MODEL

def get_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    system_prompt = (
        "You are an AI Customer Support Specialist. Use the following context to answer accurately.\n"
        "If the context doesn't contain the answer, strictly say you don't know and refer to support@company.com.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.4}
    )
    
    return create_retrieval_chain(retriever, question_answer_chain)
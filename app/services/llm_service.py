"""
LLM (Large Language Model) Service.

This file handles all interactions with the OpenAI LLM (GPT model).

What is an LLM?
- A Large Language Model is an AI that can understand and generate
  human-like text. Examples: GPT-4, GPT-4o-mini.
- We use it to generate answers to user questions based on the
  document context we provide.

How it fits in the architecture:
  Retrieved context + Question → THIS MODULE sends to LLM → Answer

Why not just use the LLM directly without RAG?
- LLMs can "hallucinate" — make up information that sounds correct.
- By providing specific document context, we ground the LLM's response
  in actual data, significantly reducing hallucinations.
- The LLM acts as a smart summarizer of the retrieved information,
  not as a source of information itself.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import OPENAI_API_KEY, LLM_MODEL, NVIDIA_BASE_URL

logger = logging.getLogger(__name__)

# --- The RAG Prompt Template ---
# This is the instruction we send to the LLM along with the context and question.
# It's carefully designed to:
# 1. Tell the LLM its role (document Q&A assistant).
# 2. Restrict it to only use the provided context (reduces hallucination).
# 3. Tell it what to do if the answer isn't in the context.
# 4. Provide the actual context and question.
RAG_PROMPT_TEMPLATE = """
You are a helpful document question-answering assistant.

Your job is to answer the user's question using ONLY the information 
contained in the provided context below. Follow these rules strictly:

1. Base your answer solely on the provided context.
2. If the answer cannot be found in the context, clearly state: 
   "I couldn't find this information in the uploaded documents."
3. Do not use any outside knowledge or make up information.
4. Be concise and direct in your answers.
5. If the context contains partial information, mention what you found 
   and note that the information may be incomplete.

Context:
{context}

Question:
{question}
"""


def get_llm() -> ChatOpenAI:
    """
    Create and return the LLM instance.
    
    Returns:
        A ChatOpenAI instance configured with our API key and model.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key is not set. "
            "Please set the OPENAI_API_KEY environment variable in your .env file."
        )
    
    return ChatOpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        temperature=0,
    )


def generate_answer(context: str, question: str) -> str:
    """
    Generate an answer using the LLM based on the provided context.
    
    This is where the "Generation" part of RAG happens:
    1. We format the prompt with the retrieved context and question.
    2. We send it to the LLM.
    3. The LLM generates an answer based only on the context.
    
    Args:
        context: The relevant document text retrieved from ChromaDB.
        question: The user's question.
        
    Returns:
        The LLM-generated answer string.
    """
    logger.info("Generating answer using LLM")
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # Get the LLM
    llm = get_llm()
    
    # Create the chain: prompt → LLM
    # The '|' operator chains the prompt and LLM together.
    # The prompt formats the template, then passes it to the LLM.
    chain = prompt | llm
    
    # Run the chain with our context and question
    response = chain.invoke({
        "context": context,
        "question": question,
    })
    
    logger.info("LLM response generated successfully")
    
    # response.content contains the actual text answer
    return response.content

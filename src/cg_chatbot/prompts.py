from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables.base import Runnable
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser


### Retrieval Grader
def retrieval_grader(model: str) -> Runnable:
    llm = ChatOllama(model=model, format="json", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a grader assessing relevance
        of a retrieved document to a user question. If the document seems related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous or irrelevant retrievals.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        Provide this binary score as JSON with the single key 'score' with no preamble or explanation. """,
            ),
            (
                "human",
                """Here is the retrieved document: {document}\n\nHere is the user question: {question}""",
            ),
        ]
    )
    return prompt | llm | JsonOutputParser()


### Generate
def generator(model: str) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an assistant for question-answering tasks. Use the following provided documents to answer
        the question. If relevant, unless the question states otherwise, we are interested only in the post-Turnover-Meeting answer. Please cite the supporting sections by section number.
        Use three sentences maximum and keep the answer concise.""",
            ),
            (
                "human",
                """Question: {question}
        Retrieved documents: {documents}""",
            ),
        ]
    )
    llm = ChatOllama(model=model, temperature=0)
    return prompt | llm | StrOutputParser()


### Hallucination Grader
def hallucination_grader(model: str) -> Runnable:
    llm = ChatOllama(model=model, format="json", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a grader assessing whether
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        single key 'score' and no preamble or explanation.""",
            ),
            (
                "human",
                """Here are the facts:
        -------
        {documents}
        -------
        Here is the answer: {generation}""",
            ),
        ]
    )
    return prompt | llm | JsonOutputParser()


### Answer Grader
def answer_grader(model: str) -> Runnable:
    llm = ChatOllama(model=model, format="json", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a grader assessing whether an
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            ),
            (
                "human",
                "Here is the answer: {generation}\n\nHere is the question: {question}",
            ),
        ]
    )
    return prompt | llm | JsonOutputParser()

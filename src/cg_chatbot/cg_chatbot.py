#!/usr/bin/env python
# coding: utf-8

# ### Tracing (optional)
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = <your-api-key>
# ```

### LLM

GRADER_LLM = "llama3.1"  # "phi3:medium-128k"
GENERATE_LLM = "llama3.1"  # "deepseek-r1:8b"
STARTING_K = 4
MAX_RUNTIME_SECONDS = 20
CHROMADB_PERSIST_DIRECTORY = "./chroma_db"
REPL_HISTFILE = ".chat_history"
REPL_HISTFILE_SIZE = 1000

from argparse import ArgumentParser, Namespace
from datetime import datetime
import os
import readline
import sys
import threading
import time
from typing_extensions import TypedDict
from typing import Any, Generator, Literal, NoReturn, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain.schema import Document


def parse_args() -> Namespace:
    parser = ArgumentParser(
        prog=sys.argv[0],
        description="Invoke an AI Chatbot that can answer questions about Cully Grove documents",
    )
    parser.add_argument(
        "--chromadb-persist-directory",
        default=CHROMADB_PERSIST_DIRECTORY,
        help="ChromaDB persist directory containing the pre-ingested docs",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=MAX_RUNTIME_SECONDS,
        help="Allow MAX_RUNTIME_SECONDS before stopping any further refinements and completing with the latest response, good or bad",
    )
    parser.add_argument("-v", "--verbose", default=False, action="store_true")

    subparsers = parser.add_subparsers(dest="mode")
    subparsers.add_parser("demo", help="Evaluate some canned questions, then quit")
    subparsers.add_parser(
        "repl", help="Provide a prompt where you can type questions interactively"
    )
    subparsers.add_parser("serve", help="spin up a web interface")

    args = parser.parse_args()
    if args.mode is None:
        print(
            f"Must specify a mode from: {list(subparsers.choices.keys())}",
            file=sys.stderr,
        )
        parser.print_usage()
        sys.exit(2)

    return args


args = parse_args()


def vprint(*print_args: str, **print_kwargs: Any):
    if args.verbose:
        print(*print_args, **print_kwargs)


def main():
    ### Index

    embedding = GPT4AllEmbeddings(
        model_name="nomic-embed-text-v1.5.f16.gguf", gpt4all_kwargs={}
    )  # type: ignore
    vectorstore = Chroma(
        collection_name="cully-grove-bylaws",
        persist_directory=args.chromadb_persist_directory,
        embedding_function=embedding,
    )

    ### Retrieval Grader

    llm = ChatOllama(model=GRADER_LLM, format="json", temperature=0)
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
    retrieval_grader = prompt | llm | JsonOutputParser()

    ### Generate

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
    llm = ChatOllama(model=GENERATE_LLM, temperature=0)
    rag_chain = prompt | llm | StrOutputParser()

    ### Hallucination Grader

    llm = ChatOllama(model=GRADER_LLM, format="json", temperature=0)
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
    hallucination_grader = prompt | llm | JsonOutputParser()

    ### Answer Grader

    llm = ChatOllama(model=GRADER_LLM, format="json", temperature=0)
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
    answer_grader = prompt | llm | JsonOutputParser()

    ### State
    # We'll implement these as a control flow in LangGraph. Define the state vector, then nodes (that can read and transform state), then edges.
    class GraphState(TypedDict, total=False):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
            retrieve_k: max number of docs to retrieve
        """

        start_time: datetime
        question: str
        generation: str
        documents: list[Document]
        retrieve_k: int

    def _get_state_or_raise(d: GraphState, key: str) -> Any:
        val = d.get(key)
        if val is None:
            raise ValueError(f"Internal logic error: expected state to have key {key}")
        return val

    ### Nodes
    def retrieve(state: GraphState) -> GraphState:
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        vprint("---RETRIEVE---")
        question = _get_state_or_raise(state, "question")
        retrieve_k = state.get("retrieve_k")
        if retrieve_k is None:  # first time around
            retrieve_k = STARTING_K
        else:
            retrieve_k += 4
        start_time = state.get("start_time") or datetime.now()

        # Retrieval
        retriever = vectorstore.as_retriever(search_kwargs={"k": retrieve_k})
        documents = retriever.invoke(question)
        return {
            "documents": documents,
            "question": question,
            "retrieve_k": retrieve_k,
            "start_time": start_time,
        }

    def generate(state: GraphState) -> GraphState:
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        vprint("---GENERATE---")
        question = _get_state_or_raise(state, "question")
        documents = _get_state_or_raise(state, "documents")

        # RAG generation
        generation = rag_chain.invoke({"documents": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state: GraphState) -> GraphState:
        """
        Determines whether the retrieved documents are relevant to the question
        If all documents are not relevant, we will set a flag to run web search.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """
        vprint("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = _get_state_or_raise(state, "question")
        documents = _get_state_or_raise(state, "documents")

        # Score each doc in parallel
        results = retrieval_grader.batch(
            [{"question": question, "document": d.page_content} for d in documents]
        )

        filtered_docs = []
        for response, d in zip(results, documents):
            grade = response["score"]
            # Document relevant
            if grade.lower() == "yes":
                vprint("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                vprint("---GRADE: DOCUMENT NOT RELEVANT---")

        return {"documents": filtered_docs, "question": question}

    ### Conditional edges

    def _time_remains(start_time: datetime) -> bool:
        now = datetime.now()
        return (now - start_time).total_seconds() < args.max_runtime_seconds

    def decide_to_generate(
        state: GraphState,
    ) -> Literal["out of time", "retrieve", "generate"]:
        """
        Determines whether to generate an answer, or retrieve more docs

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        vprint("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = _get_state_or_raise(state, "documents")
        start_time = _get_state_or_raise(state, "start_time")

        if not _time_remains(start_time):
            vprint("---DECISION: OUT OF TIME; JUST GENERATE---")
            return "out of time"

        if not filtered_documents:
            vprint("---DECISION: NO DOCUMENTS RELEVANT TO QUESTION, RETRIEVE MORE---")
            return "retrieve"
        else:
            # We have relevant documents, so generate answer
            vprint("---DECISION: GENERATE---")
            return "generate"

    ### Conditional edge
    def grade_generation_v_documents_and_question(
        state: GraphState,
    ) -> Literal["out of time", "useful", "not useful", "not supported"]:
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        vprint("---CHECK HALLUCINATIONS---")
        question = _get_state_or_raise(state, "question")
        documents = _get_state_or_raise(state, "documents")
        generation = _get_state_or_raise(state, "generation")

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]
        start_time = state.get("start_time")
        if start_time is None:
            raise ValueError("Internal state error: start_time never set")
        if not _time_remains(start_time):
            vprint("---DECISION: OUT OF TIME; GIVING LATEST RESULT---")
            return "out of time"

        # Check hallucination
        if grade == "yes":
            vprint("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            vprint("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            grade = score["score"]
            if grade == "yes":
                vprint("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                vprint("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            vprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    ### Build graph
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.set_entry_point("retrieve")
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "retrieve": "retrieve",
            "generate": "generate",
            "out of time": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "generate",
            "out of time": END,
        },
    )

    langchain_app = workflow.compile()

    # Now use it
    class Spinner:
        """
        Just a cute spinner animation to let the user know that something is happening.
        """

        busy = False
        delay = 0.1

        @staticmethod
        def next_cursor_string() -> Generator[str, Any, Any]:
            CHARS = " ⢎⡰ ⢎⡡ ⢎⡑ ⢎⠱ ⠎⡱ ⢊⡱ ⢌⡱ ⢆⡱"
            while True:
                for i in range(0, len(CHARS) // 3):
                    yield CHARS[3 * i : 3 * i + 3]

        def __init__(self, delay: Optional[float] = None):
            self.spinner_generator = self.next_cursor_string()
            if delay and float(delay):
                self.delay = delay

        def spinner_task(self):
            while self.busy:
                sys.stdout.write(next(self.spinner_generator))
                sys.stdout.write("\r")
                sys.stdout.flush()
                time.sleep(self.delay)

        def __enter__(self):
            self.busy = True
            threading.Thread(target=self.spinner_task).start()

        def __exit__(self, exception, value, tb):
            self.busy = False
            time.sleep(self.delay)
            if exception is not None:
                return False

    def eval_local(question) -> bool:
        """
        Return True if generation completed. False if generation was interrupted.
        """
        start_time = datetime.now()
        final_answer = None
        with Spinner():
            try:
                for output in langchain_app.stream({"question": question}):
                    for key, value in output.items():
                        if "generation" in value:
                            final_answer = value["generation"]
            except KeyboardInterrupt:
                print("^C Generation aborted")
                return False
        print(final_answer)
        if args.verbose:
            elapsed = datetime.now() - start_time
            print(f"---Took {elapsed.total_seconds()} seconds")
        return True

    if args.mode == "repl":

        def load_question_history():
            if os.path.exists(REPL_HISTFILE):
                readline.read_history_file(REPL_HISTFILE)

        def save_question_history():
            readline.set_history_length(REPL_HISTFILE_SIZE)
            readline.write_history_file(REPL_HISTFILE)

        def quit() -> NoReturn:
            save_question_history()
            sys.exit(0)

        load_question_history()
        while True:
            try:
                question = str(input(">>> "))
            except EOFError:  # catch Ctrl-d
                quit()
            except KeyboardInterrupt:  # catch Ctrl-c
                print()
                continue
            if question == "help":
                print(
                    "Type 'quit' to quit, otherwise just type your question, followed by Enter"
                )
            elif question == "quit":
                quit()
            elif question == "":
                continue
            else:
                eval_local(question)

    elif args.mode == "demo":
        questions = [
            "How many owners need to be present to form a quorum?",
            "How many board members can we have?",
            "Who shot Alexander Hamilton?",
            "Compose a very short poem about limited common elements.",
            "What is the capital contribution to be paid by buyers of a unit?",
            "How much notice is required to the community before a board meeting?",
            "What are the necessary conditions to record a lien against a unit?",
        ]
        for question in questions:
            print(f">>> {question}")
            completed = eval_local(question)
            if not completed:
                break
    elif args.mode == "serve":
        import subprocess
        import requests
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import HTMLResponse, JSONResponse, Response
        from starlette.routing import Route
        import uvicorn

        HEALTHCHECK_URL = "http://localhost:11434/api/show"
        HEALTHCHECK_DATA = f'{{"model":"{GRADER_LLM}"}}'

        async def homepage(request: Request) -> Response:
            html = open("index.html").read()
            return HTMLResponse(html)

        async def api(request: Request) -> Response:
            start_time = datetime.now()
            final_answer = None
            question = request.query_params["q"]
            for output in langchain_app.stream({"question": question}):
                for key, value in output.items():
                    if "generation" in value:
                        final_answer = value["generation"]
            elapsed = (datetime.now() - start_time).total_seconds()
            return JSONResponse(
                {
                    "question": question,
                    "response": final_answer,
                    "elapsed_seconds": elapsed,
                }
            )

        async def healthcheck(request: Request) -> Response:
            # Only status code is meaningful. Return empty body.
            # If we got here at all, we're healthy. Check on ollama.
            try:
                response = requests.post(HEALTHCHECK_URL, HEALTHCHECK_DATA)
                return Response("", status_code=response.status_code)
            except requests.ConnectionError:
                return Response("", status_code=500)

        async def version(request: Request) -> Response:
            commit_process = subprocess.run(
                ["/usr/bin/git", "rev-parse", "--short", "HEAD"], capture_output=True
            )
            commit = commit_process.stdout.decode("ascii").rstrip()
            clean_process = subprocess.run(["/usr/bin/git", "diff", "--quiet"])
            clean = clean_process.returncode == 0
            return JSONResponse({"version": commit, "clean": clean})

        routes = [
            Route("/", homepage),
            Route("/api/query", api),
            Route("/api/healthcheck", healthcheck),
            Route("/api/version", version),
        ]

        app = Starlette(debug=True, routes=routes)

        uvicorn.run(app, host="0.0.0.0", port=8000)

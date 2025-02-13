from datetime import datetime
from typing import Any, Literal
from typing_extensions import TypedDict

from langchain.schema import Document
from langchain_chroma import Chroma

from cg_chatbot import constants
from cg_chatbot import prompts


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


class GraphNodes:
    retrieval_grader = prompts.retrieval_grader(constants.GRADER_LLM)
    generator = prompts.generator(constants.GENERATE_LLM)
    hallucination_grader = prompts.hallucination_grader(constants.GRADER_LLM)
    answer_grader = prompts.answer_grader(constants.GRADER_LLM)

    def __init__(self, vector_store: Chroma, max_runtime_seconds, verbose=False):
        self.vector_store = vector_store
        self.max_runtime_seconds = max_runtime_seconds
        self.verbose = verbose

    def vprint(self, *print_args: str, **print_kwargs: Any):
        if self.verbose:
            print(*print_args, **print_kwargs)

    @staticmethod
    def _get_state_or_raise(d: GraphState, key: str) -> Any:
        val = d.get(key)
        if val is None:
            raise ValueError(f"Internal logic error: expected state to have key {key}")
        return val

    ### Nodes
    def retrieve(self, state: GraphState) -> GraphState:
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        self.vprint("---RETRIEVE---")
        question = self._get_state_or_raise(state, "question")
        retrieve_k = state.get("retrieve_k")
        if retrieve_k is None:  # first time around
            retrieve_k = constants.STARTING_K
        else:
            retrieve_k += 4
        start_time = state.get("start_time") or datetime.now()

        # Retrieval
        retriever = self.vector_store.as_retriever(search_kwargs={"k": retrieve_k})
        documents = retriever.invoke(question)
        return {
            "documents": documents,
            "question": question,
            "retrieve_k": retrieve_k,
            "start_time": start_time,
        }

    def generate(self, state: GraphState) -> GraphState:
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        self.vprint("---GENERATE---")
        question = self._get_state_or_raise(state, "question")
        documents = self._get_state_or_raise(state, "documents")

        # RAG generation
        generation = self.generator.invoke(
            {"documents": documents, "question": question}
        )
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Determines whether the retrieved documents are relevant to the question
        If all documents are not relevant, we will set a flag to run web search.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """
        self.vprint("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = self._get_state_or_raise(state, "question")
        documents = self._get_state_or_raise(state, "documents")

        # Score each doc in parallel
        results = self.retrieval_grader.batch(
            [{"question": question, "document": d.page_content} for d in documents]
        )

        filtered_docs = []
        for response, d in zip(results, documents):
            grade = response["score"]
            # Document relevant
            if grade.lower() == "yes":
                self.vprint("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                self.vprint("---GRADE: DOCUMENT NOT RELEVANT---")

        return {"documents": filtered_docs, "question": question}

    ### Conditional edges
    def _time_remains(self, start_time: datetime) -> bool:
        now = datetime.now()
        return (now - start_time).total_seconds() < self.max_runtime_seconds

    def decide_to_generate(
        self,
        state: GraphState,
    ) -> Literal["out of time", "retrieve", "generate"]:
        """
        Determines whether to generate an answer, or retrieve more docs

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        self.vprint("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = self._get_state_or_raise(state, "documents")
        start_time = self._get_state_or_raise(state, "start_time")

        if not self._time_remains(start_time):
            self.vprint("---DECISION: OUT OF TIME; JUST GENERATE---")
            return "out of time"

        if not filtered_documents:
            self.vprint(
                "---DECISION: NO DOCUMENTS RELEVANT TO QUESTION, RETRIEVE MORE---"
            )
            return "retrieve"
        else:
            # We have relevant documents, so generate answer
            self.vprint("---DECISION: GENERATE---")
            return "generate"

    ### Conditional edge
    def grade_generation_v_documents_and_question(
        self,
        state: GraphState,
    ) -> Literal["out of time", "useful", "not useful", "not supported"]:
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        self.vprint("---CHECK HALLUCINATIONS---")
        question = self._get_state_or_raise(state, "question")
        documents = self._get_state_or_raise(state, "documents")
        generation = self._get_state_or_raise(state, "generation")

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]
        start_time = state.get("start_time")
        if start_time is None:
            raise ValueError("Internal state error: start_time never set")
        if not self._time_remains(start_time):
            self.vprint("---DECISION: OUT OF TIME; GIVING LATEST RESULT---")
            return "out of time"

        # Check hallucination
        if grade == "yes":
            self.vprint("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            self.vprint("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            grade = score["score"]
            if grade == "yes":
                self.vprint("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                self.vprint("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            self.vprint(
                "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---"
            )
            return "not supported"

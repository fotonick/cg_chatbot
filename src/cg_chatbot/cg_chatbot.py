#!/usr/bin/env python
# coding: utf-8

# ### Tracing (optional)
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = <your-api-key>
# ```


from argparse import ArgumentParser, Namespace
import sys

from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph

from cg_chatbot import constants
from cg_chatbot import endpoints
from cg_chatbot import graph_nodes
from cg_chatbot import termutils


def parse_args() -> Namespace:
    parser = ArgumentParser(
        prog=sys.argv[0],
        description="Invoke an AI Chatbot that can answer questions about Cully Grove documents",
    )
    parser.add_argument(
        "--chromadb-persist-directory",
        default=constants.CHROMADB_PERSIST_DIRECTORY,
        help="ChromaDB persist directory containing the pre-ingested docs",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=constants.MAX_RUNTIME_SECONDS,
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


def main():
    args = parse_args()

    ### Index
    embedding = GPT4AllEmbeddings(
        model_name="nomic-embed-text-v1.5.f16.gguf", gpt4all_kwargs={}
    )  # type: ignore
    vectorstore = Chroma(
        collection_name="cully-grove-bylaws",
        persist_directory=args.chromadb_persist_directory,
        embedding_function=embedding,
    )

    ### Build graph
    nodes = graph_nodes.GraphNodes(vectorstore, args.max_runtime_seconds, args.verbose)
    workflow = StateGraph(graph_nodes.GraphState)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.set_entry_point("retrieve")
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        nodes.decide_to_generate,
        {
            "retrieve": "retrieve",
            "generate": "generate",
            "out of time": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        nodes.grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "generate",
            "out of time": END,
        },
    )
    langchain_app = workflow.compile()

    # Now use it
    eval_local = termutils.make_eval_local(langchain_app, args.verbose)

    if args.mode == "repl":
        termutils.load_question_history()
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
        from pprint import pprint
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn

        api = endpoints.API(langchain_app)
        routes = [
            Route("/", api.homepage),
            Route("/api/query", api.query),
            Route("/api/healthcheck", api.healthcheck),
            Route("/api/version", api.version),
        ]
        pprint(routes)
        app = Starlette(debug=True, routes=routes)
        uvicorn.run(app, host="0.0.0.0", port=8000)

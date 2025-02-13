GRADER_LLM = "llama3.1"  # "phi3:medium-128k"
GENERATE_LLM = "llama3.1"  # "deepseek-r1:8b"
STARTING_K = 4
MAX_RUNTIME_SECONDS = 20
CHROMADB_PERSIST_DIRECTORY = "./chroma_db"
HEALTHCHECK_URL = "http://localhost:11434/api/show"
HEALTHCHECK_DATA = f'{{"model":"{GRADER_LLM}"}}'

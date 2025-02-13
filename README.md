Cully Grove Chatbot
===================

# Goal
1. Create a chatbot that we can ask questions about the [Cully Grove Bylaws](https://cullygrove.org/wp-content/uploads/2011/04/cully-grove-declaration-and-bylaws-recorded.pdf) that can run on my laptop.
2. (Optional, TODO) Extend the chatbot's knowledge to documents in Cully Grove's Google Drive shared drive.

# Strategy
* OCR [Cully Grove Bylaws PDF](https://cullygrove.org/wp-content/uploads/2011/04/cully-grove-declaration-and-bylaws-recorded.pdf) to Markdown using [Marker](https://github.com/VikParuchuri/marker)
* Do a cursory cleanup / hand-edit of the Markdown
* Heavily adapt the [Langchain example for a RAG agent](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_rag_agent_llama3_local.ipynb). I specifically removed web search as a supplemental source of facts.
* Use [llama3.1:8b](https://ollama.com/library/llama3.1) served via [Ollama](https://ollama.com), as a fairly modern model that can run locally.

## Status

The chatbot answers questions on the bylaws correctly most of the time and is approximately fast enough (~13 s per generation) on my MacBook Pro (M1 Pro). All the code after ingestion is in a single file. Yes, I do know better, but no, I'm not planning on fixing it in the near term; PRs welcome.

# Setup

To run at a reasoanble speed, you need either a high-spec M-series Mac or else a GPU with several GB of VRAM.

1. Install LLM
    1. Install [Ollama](https://ollama.com)
    2. (optional small optimization; varies by platform) Set up Ollama to start up with the environment variable `OLLAMA_NUM_PARALLEL` set to `4` or greater
    3. `ollama pull llama3.1`
2. Install program dependencies
    1. Install [uv](https://github.com/astral-sh/uv), which allows you to reproduce my environment very closely.
    2. On non-Macs, download a recent version of the NVIDIA CUDA Toolkit, version 12.5 at the time of this writing
    3. `uv sync` to install Python dependencies.
3. Build vector store from source documents: `uv run ingest`

# Running

The primary way to run the chatbot locally is with `uv run cg_chatbot repl`, which will give you a prompt where you can ask questions. Don't forget to `conda activate $MY_ENV`. You can retrieve previously asked questions, persistent across sessions, with the ↑ and ↓ arrow keys.

```
>>> How many owners need to be present to form a quorum?
According to the provided documents, a quorum is formed when the presence of owners holding 51 percent or more of the voting power of the Association is present at any meeting (Section 3.11). This means that a majority of the owners must be present in person or by proxy to constitute a quorum.
>>> How many board members can we have?
 ⢎⠱
```

There's also `uv run cg_chatbot.py demo`, which will run a pre-programmed list of questions as a demonstration. It was useful during development to give me a pulse on performance, particularly with the `-v` option to print timings.

Finally, there's the web interface, which you can invoke with `uv run cg_chatbot.py serve`. Browse to http://localhost:8000 to enter your questions.

# Installing a systemd service

On a Linux server with systemd, you can set the web interface to run as a service that is started when the machine starts, even before you log in.

1. Edit `cg_chatbot.service` with your own username and home directory.
2. `sudo cp cg_chatbot.service /etc/systemd/system`
3. `sudo service cg_chatbot start`

You can check the status with `service cg_chatbot status` or jump to the bottom of `less /var/log/syslog`. The latter is useful for seeing error messages during debug.


License
=======

This software is released under the MIT license.

Contribution
============

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion into this project shall be licensed as MIT, without any additional terms or conditions.

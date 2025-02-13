from datetime import datetime
import requests
import subprocess

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from cg_chatbot import constants


class API:
    def __init__(self, app):
        self.app = app

    async def homepage(self, request: Request) -> Response:
        html = open("index.html").read()
        return HTMLResponse(html)

    async def query(self, request: Request) -> Response:
        start_time = datetime.now()
        final_answer = None
        question = request.query_params["q"]
        for output in self.app.stream({"question": question}):
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

    async def healthcheck(self, request: Request) -> Response:
        # Only status code is meaningful. Return empty body.
        # If we got here at all, we're healthy. Check on ollama.
        try:
            response = requests.post(
                constants.HEALTHCHECK_URL, constants.HEALTHCHECK_DATA
            )
            return Response("", status_code=response.status_code)
        except requests.ConnectionError:
            return Response("", status_code=500)

    async def version(self, request: Request) -> Response:
        commit_process = subprocess.run(
            ["/usr/bin/git", "rev-parse", "--short", "HEAD"], capture_output=True
        )
        commit = commit_process.stdout.decode("ascii").rstrip()
        clean_process = subprocess.run(["/usr/bin/git", "diff", "--quiet"])
        clean = clean_process.returncode == 0
        return JSONResponse({"version": commit, "clean": clean})

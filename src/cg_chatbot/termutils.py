import os
import readline
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Generator, NoReturn, Optional

REPL_HISTFILE = ".chat_history"
REPL_HISTFILE_SIZE = 1000


def load_question_history():
    if os.path.exists(REPL_HISTFILE):
        readline.read_history_file(REPL_HISTFILE)


def save_question_history():
    readline.set_history_length(REPL_HISTFILE_SIZE)
    readline.write_history_file(REPL_HISTFILE)


def quit() -> NoReturn:
    save_question_history()
    sys.exit(0)


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


def make_eval_local(app, verbose=False) -> Callable[[str], bool]:
    def eval_local(question: str) -> bool:
        """
        Return True if generation completed. False if generation was interrupted.
        """
        start_time = datetime.now()
        final_answer = None
        with Spinner():
            try:
                for output in app.stream({"question": question}):
                    for key, value in output.items():
                        if "generation" in value:
                            final_answer = value["generation"]
            except KeyboardInterrupt:
                print("^C Generation aborted")
                return False
        print(final_answer)
        if verbose:
            elapsed = datetime.now() - start_time
            print(f"---Took {elapsed.total_seconds()} seconds")
        return True

    return eval_local

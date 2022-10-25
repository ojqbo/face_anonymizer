#!/usr/bin/python3
# from jsmin import jsmin
# from pathlib import Path
import asyncio
import string
import random
import logging

# def make_single_page_app(DEV_WEBSITE_ROOT_PATH, INDEX_HTML_PATH):
#     """prepares a single html file from index.html and main.js files
#     source files should be placed in DEV_WEBSITE_ROOT_PATH
#     the final .html file with embedded script will be saved into INDEX_HTML_PATH
#     """
#     jspath = Path(DEV_WEBSITE_ROOT_PATH) / Path("main.js")
#     html_in_path = Path(DEV_WEBSITE_ROOT_PATH) / Path("index.html")
#     html_out_path = Path(INDEX_HTML_PATH)
#     with open(jspath) as f:
#         script = jsmin(f.read(), quote_chars="'\"`")
#         #script = f.read()
#     with open(html_in_path) as f:
#         html_page = f.read()
#     with open(html_out_path, "w") as f:
#         f.write(html_page.replace('<script src="main.js"></script>',f"<script>{script}</script>"))


def unique_id_generator(size=22, chars=string.ascii_letters + string.digits):
    """generates random identifier. If default args are used,
    it is save to assume that no two ids will ever repeat.
    Cardinality of possible outcomes is len(chars)**size.
    Defaults to 62**22 (130.99 bits of 'randomness')

    Args:
        size (int, optional): length of returned random id string. Defaults to 22.
        chars (str, optional): set of characters to choose from. Defaults to string.ascii_letters+string.digits.

    Returns:
        str: string of randomly chosen `chars` of length `size`
    """
    # https://stackoverflow.com/a/2257449
    return "".join(random.choice(chars) for _ in range(size))


def catch_background_task_exception(task: asyncio.Task) -> None:
    # https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task cancellation should not be logged as an error.
    except Exception:
        logging.exception(f"Exception raised by task = {task}")

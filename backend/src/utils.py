import asyncio
import logging
import random
import string


def unique_id_generator(size=22, chars=string.ascii_letters + string.digits):
    """generates random identifier. If default args are used,
    it is save to assume that no two ids will ever repeat.
    Cardinality of possible outcomes is len(chars)**size.
    Defaults to 62**22 (130.99 bits of 'randomness')

    Args:
        size (int, optional): length of returned random id string. Defaults to 22.
        chars (str, optional): set of characters to choose from.
            Defaults to string.ascii_letters+string.digits

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

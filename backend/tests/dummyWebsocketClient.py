import json
from pathlib import Path


class dummyWebsocketClient:
    def __init__(self, input_file_path: str | Path):
        with open(input_file_path, "rb") as f:
            self._filebytes = f.read()
        self._size = len(self._filebytes)
        self._filename = Path(input_file_path).name
        self.received_messages: list[str] = []

    def msg_file_available(self):
        return json.dumps(
            {
                "msg": "file available",
                "name": str(self._filename),
                "size": self._size,
                "type": "video/mp4",
            }
        )

    def msg_get(self, S: int, E: int):
        return json.dumps(
            {
                "msg": "get",
                "from": S,
                "upto": E,
            }
        )

    def msg_download(self, config: dict = {}):
        return json.dumps({"msg": "user config, request download", **config})

    async def send_str(self, msg: str):
        self.msg_receive(msg)

    def msg_receive(self, msg):
        self.received_messages.append(msg)

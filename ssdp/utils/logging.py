from __future__ import annotations

import loguru


class OnceFilter:
    def __init__(self, bind_key: str = "once") -> None:
        self.bind_key = bind_key
        self.messages = set()

    def __call__(self, record: loguru.Record) -> bool:
        if record["extra"].get(self.bind_key):
            message = record["message"]
            if message in self.messages:
                return False
            self.messages.add(message)
        return True

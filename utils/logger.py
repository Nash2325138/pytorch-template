import json
import logging


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%m-%d %H:%M:%S',
)


class Logger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """

    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)

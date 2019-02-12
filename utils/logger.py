import json
import sys
import logging


stream_handler = logging.StreamHandler(sys.stdout)
format_ = ('[%(asctime)s] {%(filename)s:%(lineno)d} '
           '%(levelname)s - %(message)s')

try:
    # use colored logs if installed
    import coloredlogs
    formatter = coloredlogs.ColoredFormatter(fmt=format_)
    stream_handler.setFormatter(formatter)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format=format_,
    datefmt='%m-%d %H:%M:%S',
    handlers=[stream_handler]
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

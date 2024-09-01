import logging
import os
import sys

from dotenv import load_dotenv

PATH_FILE = "./data/Be_Good.pdf"


def load_configurations():
    load_dotenv()
    return os.getenv("API_KEY_OPENAI")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

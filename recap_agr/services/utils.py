from __future__ import absolute_import, annotations

import logging
import sys
from typing import Any, Dict, Generator, List

import nltk
from ruamel import yaml

logger = logging.getLogger("recap")


class Config:
    """Store general application settings as a singleton"""

    # Here will be the instance stored.
    _instance = None
    _config: Dict[str, Any]
    _config_error = "The key is not defined in 'config.yml'. This is most likely caused by an old version of that file. Look at 'config_example.yml' to see all options."

    @staticmethod
    def get_instance() -> Config:
        """ Static access method. """
        if Config._instance is None:
            Config()
        return Config._instance # pyright: ignore

    def __init__(self):
        """ Virtually private constructor. """
        if Config._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Config._instance = self
            with open("config.yml", "r") as f:
                self._config = yaml.safe_load(f)

    def items(self):
        return self._config.items()

    def __getitem__(self, key: str):
        if key in self._config:
            return self._config[key]
        else:
            raise ValueError(
                f"Accessing config '{key}' not possible.", self._config_error
            )

    def __setitem__(self, key: str, value: Any):
        if key in self._config:
            self._config[key] = value
        else:
            raise ValueError(
                f"Accessing config '{key}' not possible.", self._config_error
            )


def preprocess_text(text: str) -> str:
    # TODO: Check for german texts
    # config = Config.get_instance()
    # out = text.lower() if config["lowercase"] else text

    # return out.translate(get_umlauts_map())
    return text


def get_tokens(text: str) -> List[str]:
    """Split a string into a set of unique tokens"""

    config = Config.get_instance()
    text_normalized = preprocess_text(text)
    tokens = nltk.word_tokenize(text_normalized, config["language"])

    if config["ignore_stopwords"]:
        stopwords = nltk.corpus.stopwords.words(config["language"])
        tokens = [word for word in tokens if word not in stopwords]

    if config["stemming"]:
        stemmer = get_stemmer()
        tokens = [stemmer.stem(word) for word in tokens]

    return tokens


def get_stemmer():
    config = Config.get_instance()
    return (
        nltk.stem.snowball.GermanStemmer()
        if config["language"] == "german"
        else nltk.stem.snowball.EnglishStemmer()
    )


def get_umlauts_map():
    """Get a table for converting the German umlauts using the str.translate() func."""

    umlaut_dict = {
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
    }

    return {ord(key): val for key, val in umlaut_dict.items()}


def print_progress(iteration, total, prefix="", suffix="", decimals=1, bar_length=100):
    """Call in a loop to create terminal progress bar"""

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "█" * filled_length + "—" * (bar_length - filled_length)

    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)), # pyright: ignore

    if iteration + 1 == total:
        sys.stdout.write("\n")

    sys.stdout.flush()


def generate_id(current_id: int = 1001) -> Generator:
    """Generate unique id

    Create new id by calling next()
    """

    while True:
        yield current_id
        current_id += 1

from abc import ABC, abstractmethod

from llmeter.endpoints.base import InvocationResponse
from llmeter.results import Result
from llmeter.runner import Runner

class Callback(ABC):

    def before_invoke(self, payload: dict):
        pass

    def after_invoke(self, response: InvocationResponse):
        pass

    def before_run(self, runner: Runner):
        pass

    def after_run(self, result: Result):
        pass

    @abstractmethod
    def save_to_file(self)-> str|None:
        pass

    @staticmethod
    def load_from_file(path: str)->'Callback':
        # check if it's one of built-in modules
        # use the _load_from_file method to load the configuration
        pass

    @classmethod
    @abstractmethod
    def _load_from_file(cls, path: str):
        pass


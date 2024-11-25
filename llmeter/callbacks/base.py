from abc import ABC, abstractmethod

from ..endpoints.base import InvocationResponse
from ..results import Result
# from ..runner import Runner

class Callback(ABC):

    async def before_invoke(self, payload: dict):
        pass

    async def after_invoke(self, response: InvocationResponse):
        pass

    async def before_run(self, runner: 'llmeter.runner.Runner'):
        pass

    async def after_run(self, result: Result):
        pass

    @abstractmethod
    def save_to_file(self)-> str|None:
        pass

    @staticmethod
    def load_from_file(path: str):
        # check if it's one of built-in modules
        # use the _load_from_file method to load the configuration
        pass

    @classmethod
    @abstractmethod
    def _load_from_file(cls, path: str):
        pass


from abc import ABC, abstractmethod

class BaseTool(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def execute(self, **kwargs):
        pass

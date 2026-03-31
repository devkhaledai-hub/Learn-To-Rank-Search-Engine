from abc import ABC, abstractmethod


class BasePipeline(ABC):

    @abstractmethod
    def load_data(self, file_path):
        pass

    @abstractmethod
    def preprocess(self, X):
        pass

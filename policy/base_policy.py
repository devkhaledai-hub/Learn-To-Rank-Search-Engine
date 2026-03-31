from abc import ABC, abstractmethod


class BasePolicy(ABC):

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass

    @abstractmethod
    def save(self, model, path_to_model):
        pass

    @abstractmethod
    def load(self, path_to_model):
        pass

from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def train(self, X_train, y_train, groups_train=None,
              X_val=None, y_val=None, groups_val=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

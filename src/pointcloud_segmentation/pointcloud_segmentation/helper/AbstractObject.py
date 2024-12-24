from abc import ABC, abstractmethod

class AbstractObject(ABC):
         
    @abstractmethod
    def get_depth(self) -> float:
        return -1.0
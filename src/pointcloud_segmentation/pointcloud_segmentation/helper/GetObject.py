from .AbstractObject import AbstractObject
from .Person import Person
from .Dog import Dog

class GetObject():
        
    def createObject(self, class_name: str) -> AbstractObject:
        if class_name == "person":
            return Person()
        elif class_name == "dog":
            return Dog()
        else:
            return None
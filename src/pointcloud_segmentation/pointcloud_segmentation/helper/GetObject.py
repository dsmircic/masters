from .AbstractObject    import AbstractObject
from .Person            import Person
from .Dog               import Dog
from .Bottle            import Bottle
from .Umbrella          import Umbrella
from .Chair             import Chair

class GetObject():
        
    def createObject(self, class_name: str) -> AbstractObject:
        if class_name == "person":
            return Person()
        elif class_name == "dog":
            return Dog()
        elif class_name == "bottle":
            return Bottle()
        elif class_name == "umbrella":
            return Umbrella()
        elif class_name == "chair":
            return Chair()
        else:
            return None
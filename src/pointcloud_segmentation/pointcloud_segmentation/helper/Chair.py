from .AbstractObject import AbstractObject

class Chair(AbstractObject):
    def __init__(self):
        super().__init__()
        
    def get_depth(self):
        return 0.3
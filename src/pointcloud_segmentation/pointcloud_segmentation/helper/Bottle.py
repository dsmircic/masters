from .AbstractObject import AbstractObject

class Bottle(AbstractObject):
    def __init__(self):
        super().__init__()
        
    def get_depth(self):
        return 0.2
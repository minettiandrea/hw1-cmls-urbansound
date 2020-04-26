from .sound import Sound 

class FeatureExtraction:

    __sound = None

    def __init__(self,sound:Sound):
        self.__sound = sound

    def extract(self):
        """
        TODO

        Extract features using MFCC
        """

import csv
from lib.sound import Sound


class SoundClass:
    name = ""
    """Name of the class of sound"""

    positive = []
    """
    Positive cases of this class in the dataset.
    List of lib.sound.Sound
    """

    negative = []
    """
    Negative cases of this class in the dataset
    List of lib.sound.Sound
    """

    def __init__(self,name,positive,negative):
        self.name = name
        self.positive = positive
        self.negative = negative

class Metadata:

    __metadata = []

    def __init__(self,path: str):
        self.__load(path)

    #load metadata from CSV, call it before any other methods
    def __load(self, path):
        with open(path, newline='') as metadatacsv:
            reader = csv.DictReader(metadatacsv)
            for row in reader:
                self.__metadata.append(row)
    
    def train_set(self, exclude_folder: int):
        """
        Extract the training set excluding the folder number `exclude_folder`

        It returns a dictionary with for each class the list of sound in the trainig set.
        Sound are encapsulated in a convenience class Sound

        Parameters
        ---------- 
        exculde_folder: int
            folder to exclude for this training set

        Returns
        -------
            List of lib.metadata.SoundClass

        """
        selected_rows = [ row for row in self.__metadata if row['fold'] != str(exclude_folder)]
        
        def getClass(cls):
            positive = list(map(lambda x: Sound(x), [row for row in selected_rows if row['class'] == cls]))
            negative = list(map(lambda x: Sound(x), [row for row in selected_rows if row['class'] != cls]))
            return SoundClass(cls,positive,negative)

        return list(map(lambda c: getClass(c),self.classes()))


    def test_set(self, folder: int):
        """
        Extract the test set from the selected folder.
        Useful for testing set

        Parameters
        ---------- 
        folder: int
            folder of the test set

        Returns
        -------
            List of lib.metadata.SoundClass
        
        """
        selected_rows =[ row for row in self.__metadata if row['fold'] == str(folder)]

        def getClass(cls):
            positive = list(map(lambda x: Sound(x), [row for row in selected_rows if row['class'] == cls]))
            negative = list(map(lambda x: Sound(x), [row for row in selected_rows if row['class'] != cls]))
            return SoundClass(cls,positive,negative)

        return list(map(lambda c: getClass(c),self.classes()))

    def classes(self):
        """
        Return the set of classes present in metadata
        """
        return set(map(lambda x: x['class'], self.__metadata))
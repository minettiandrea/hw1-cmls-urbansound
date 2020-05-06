
import csv
from lib.sound import Sound, FeatureExtractionParameters
from joblib import Parallel, delayed
from tqdm import tqdm   #to monitor loops during computation

class SoundClass:
    name = ""
    """Name of the class of sound"""

    positive = []
    """
    Positive cases of this class in the dataset.
    List of lib.sound.Sound
    """

    def __init__(self,name,positive):
        self.name = name
        self.positive = positive

class Metadata:

    __metadata = []

    def __init__(self,path: str):
        self.__load(path)

    #load metadata from CSV, call it before any other methods
    def __load(self, path):
        with open(path, newline='') as metadatacsv:
            reader = csv.DictReader(metadatacsv)
            for row in reader:
                self.__metadata.append(Sound(row))

    def calculate_all_features(self,params:FeatureExtractionParameters):
        def process(s:Sound):
            s.feature_extraction(params)
        
        # n_jobs=1 means: use all available cores
        Parallel(n_jobs=-1, backend='threading', verbose=10)(delayed(process)(node) for node in self.__metadata)
    
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
        selected_rows = [ row for row in self.__metadata if row.folder != str(exclude_folder)]
        
        def getClass(cls):
            positive = [row for row in selected_rows if row.sound_class == cls]
            return SoundClass(cls,positive)

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
        selected_rows =[ row for row in self.__metadata if row.folder == str(folder)]

        def getClass(cls):
            positive = [row for row in selected_rows if row.sound_class == cls]
            return SoundClass(cls,positive)

        return list(map(lambda c: getClass(c),self.classes()))

    def classes(self):
        """
        Return the set of classes present in metadata
        """
        return set(map(lambda x: x.sound_class, self.__metadata))
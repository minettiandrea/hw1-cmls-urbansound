import librosa
import numpy as np

class Sound:

    sound_class = ""
    """Class of the current sound"""

    __file = ""
    __start = 0.0
    __end = 0.0

    def __init__(self,row):
        self.__file = "UrbanSound8K/audio/fold" + row['fold'] + "/" + row['slice_file_name']
        self.sound_class = row['class']
        self.__start = float(row['start'])
        self.__end = float(row['end']) 

    def __str__(self):
        return self.__file + " of class " + self.sound_class

    def __repr__(self):
        return self.__str__()

    def __duration(self):
        return self.__end - self.__start

    def feature_extraction(self):
        """Extract the features of this sample"""
        y, sr = self.load()
        mfcc_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 13)
        return np.mean(mfcc_matrix, axis=1)

    def load(self):
        """
        Load the sample and retun the vector representing the sample and his sample rate

        Returns
        -------
        y    : np.ndarray [shape=(n,) or (2, n)]
            audio time series

        sr   : number > 0 [scalar]
            sampling rate of `y`
        """
        #print(self.__file)

        #No need to cut, they are alredy cutted from the original file, see UrbanSound8K_README.txt
        return librosa.load(self.__file, sr=None) 
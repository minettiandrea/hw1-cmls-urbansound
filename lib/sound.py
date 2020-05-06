import librosa
import numpy as np

class FeatureExtractionParameters:
    hop_length:int
    n_mfcc:int

    def __init__(self,hop_length=None, n_mfcc=None):
        self.hop_length = hop_length or 1024
        self.n_mfcc = n_mfcc or 13
        pass

    def __eq__(self, other):
        if other is None:
            return False
        return self.hop_length == other.hop_length and self.n_mfcc == other.n_mfcc

    def __str__(self):
        return "hop length:" + str(self.hop_length) + " n_mfcc:" + str(self.n_mfcc)


class Sound:

    sound_class = ""
    """Class of the current sound"""

    folder = None

    __file = ""
    __start = 0.0
    __end = 0.0

    __features_params = None
    __features = None

    def __init__(self,row):
        self.__file = "UrbanSound8K/audio/fold" + row['fold'] + "/" + row['slice_file_name']
        self.sound_class = row['class']
        self.folder = row['fold']
        self.__start = float(row['start'])
        self.__end = float(row['end']) 

    def __str__(self):
        return self.__file + " of class " + self.sound_class

    def __repr__(self):
        return self.__str__()

    def __duration(self):
        return self.__end - self.__start

    def feature_extraction(self, params:FeatureExtractionParameters):
        """Extract the features of this sample"""
        if self.__features is not None and self.__features_params == params:
            return self.__features
        self.__features_params = params
        y, sr = self.load()
        #MFCCS
        mfcc_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = params.n_mfcc, hop_length=params.hop_length)
        #Chroma features
        chroma_matrix = librosa.feature.chroma_stft(y=y, sr=sr)
        #Zero Crossing Rate
        ZCR = librosa.feature.zero_crossing_rate(y=y)
        #Spectral Centroid
        SC = librosa.feature.spectral_centroid(y=y, sr=sr)
        #Spectral Bandwidth
        SB = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        #Root Mean Square
        RMS = librosa.feature.rms(y=y)
        #Saving all the features
        self.__features = np.concatenate((np.mean(mfcc_matrix, axis=1), np.mean(chroma_matrix, axis=1), np.mean(ZCR, axis=1), np.mean(SC, axis=1), np.mean(SB, axis=1), np.mean(RMS, axis=1)), axis=0)
        return self.__features

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
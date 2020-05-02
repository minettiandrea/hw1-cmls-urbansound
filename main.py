from lib.metadata import Metadata
from lib.metadata import SoundClass
from lib.sound import Sound
import numpy as np


metadata = Metadata('UrbanSound8K/metadata/UrbanSound8K.csv')

#get metadata train set excluding first directory
sound_classes = metadata.train_set(1)

def forEachClass(c: SoundClass): 
    return np.array(list(map(lambda s: s.feature_extraction(),c.positive[:5]))) #Test run, take only the firsts 5 sounds of each class
    #return np.array(list(map(lambda s: s.feature_extraction(),c.positive))) #Full run, take all sounds

X = np.array(list(map(forEachClass,sound_classes)))
Y = np.array(list(map(lambda x: x.name,sound_classes)))

print(X)

test_set = metadata.test_set(1)

def predict(s:Sound):
    return "predicted sound_class"

def forEachTestSound(s:Sound):
    print(s.sound_class + " is equals to " + predict(s) + "?")
    pass

test_result = np.array(list(map(forEachTestSound,test_set[:10]))) 
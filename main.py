from lib.metadata import Metadata
from lib.metadata import SoundClass
from lib.sound import Sound
from tqdm import tqdm   #to monitor loops during computation
import numpy as np
import sklearn as skl
import scikitplot as skplt
import matplotlib.pyplot as plt


metadata = Metadata('UrbanSound8K/metadata/UrbanSound8K.csv')

#get metadata train set excluding first directory
train_set = metadata.train_set(1)
#get metadata test set for the selected folder
test_set = metadata.test_set(1)

#extract features for each sound that belong to class c
def forEachClass(c: SoundClass): 
    return np.array(list(map(lambda s: s.feature_extraction(), tqdm(c.positive)))) #Test run, take only the firsts 5 sounds of each class
    #return np.array(list(map(lambda s: s.feature_extraction(),c.positive))) #Full run, take all sounds

X_train = np.array(list(map(forEachClass, train_set)))    #extract the training set features
X_train = np.concatenate(X_train, axis=0)
y_train = np.array(list(map(lambda x: [x.name]*len(x.positive), train_set)))    #extract the training set targets (classes)
y_train = np.concatenate(y_train, axis=0)

X_test = np.array(list(map(forEachClass, test_set)))  #extract the test set features for each class
X_test = np.concatenate(X_test, axis=0)
y_test = np.array(list(map(lambda x: [x.name]*len(x.positive), test_set)))  #extract the test set targets (classes)
y_test = np.concatenate(y_test, axis=0)

#normalize the feature
scaler = skl.preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = skl.svm.SVC(C=1, kernel='rbf')     #define the model
model.fit(X_train, y_train)                 #train the model with the training set
predictions = model.predict(X_test)         #make predictions on test set

#model = skl.neural_network.MLPClassifier()

acc = skl.metrics.accuracy_score(y_test, predictions)      #compute the accuracy
print(f'\nAccuracy: {acc}')
skplt.metrics.plot_confusion_matrix(y_test, predictions)   #compute the confusion matrix
plt.show()
from lib.metadata import Metadata
from lib.metadata import SoundClass
from lib.sound import Sound,FeatureExtractionParameters
from tqdm import tqdm   #to monitor loops during computation
import numpy as np
import sklearn as skl
import scikitplot as skplt
import matplotlib.pyplot as plt
import os
import datetime
import csv
from multiprocessing import freeze_support
from joblib import Parallel, delayed

class ModelParameters:
    C:float
    gamma:float

    def __init__(self,C=None, gamma=None):
        self.C = C or 1.0
        self.gamma = gamma or 'scale'
        pass

    def __str__(self):
        return "C:" + str(self.C) + " gamma:" + str(self.gamma)


def main():
    run_id = "run_"+os.environ.get('USERNAME')+"_"+datetime.datetime.now().strftime("%Y%m%d_%H%M")

    metadata = Metadata('UrbanSound8K/metadata/UrbanSound8K.csv')

    accuracies = []

    model_params = []
    for i in [0.01, 0.1, 1.0, 10, 100]:
        for j in [0.0001, 0.001, 0.01, 0.1, 1.0, 'scale', 'auto']:
            model_params.append(ModelParameters(C=i, gamma=j))
    
    def run_for_params(model_p:ModelParameters):
        
        fe_params = FeatureExtractionParameters(hop_length=1024, n_mfcc=25)

        metadata.calculate_all_features(fe_params) 

        model = skl.svm.SVC(C=model_p.C, gamma=model_p.gamma) 

        def run_folder(folder):
            #get metadata train set excluding first directory
            train_set = metadata.train_set(folder)
            #get metadata test set for the selected folder
            test_set = metadata.test_set(folder)

            #extract features for each sound that belong to class c
            def forEachClass(c: SoundClass): 
                return np.array(list(map(lambda s: s.feature_extraction(fe_params), tqdm(c.positive)))) 

            X_train = np.array(list(map(forEachClass, train_set)))    #extract the training set features
            X_train = np.concatenate(X_train, axis=0)
            y_train = np.array(list(map(lambda x: [x.name]*len(x.positive), train_set)))    #extract the training set targets (classes)
            y_train = np.concatenate(y_train, axis=0)

            X_test = np.array(list(map(forEachClass, test_set)))  #extract the test set features for each class
            X_test = np.concatenate(X_test, axis=0)
            y_test = np.array(list(map(lambda x: [x.name]*len(x.positive), test_set)))  #extract the test set targets (classes)
            y_test = np.concatenate(y_test, axis=0)

            #normalize the features
            scaler = skl.preprocessing.RobustScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            #model = skl.svm.SVC(C=1, kernel='rbf')     #define the model
            model.fit(X_train, y_train)                 #train the model with the training set
            predictions = model.predict(X_test)         #make predictions on test set

            acc = skl.metrics.accuracy_score(y_test, predictions)      #compute the accuracy
            accuracies.append({"name": str(model_p), "acc": acc, "folder": folder})
            print(f'\nAccuracy: {acc}')

            skplt.metrics.plot_confusion_matrix(y_test, predictions)   #compute the confusion matrix
            plt.savefig('runs/'+run_id+'_folder_'+str(folder)+'_hs_'+str(model_p.C)+'_mfcc_'+str(model_p.gamma)+'.png')

        for folder in range(1, 11):
            run_folder(folder)

    for model_p in model_params:
        run_for_params(model_p)

    with open('runs/'+run_id + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Folder','Name','Accuracy'])
        for acc in accuracies:
            writer.writerow([acc['folder'],acc['name'],acc['acc']])


if __name__ == "__main__":
    freeze_support()   # required to use multiprocessing
    main()
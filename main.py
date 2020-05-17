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
from statistics import mean 

def main():
    run_id = "run_"+os.environ.get('USERNAME')+"_"+datetime.datetime.now().strftime("%Y%m%d_%H%M")
    metadata = Metadata('UrbanSound8K/metadata/UrbanSound8K.csv',limit = -1)
    accuracies = []
    folder_accuracy = []
    y_all = []
    predictions_all = []
    fe_params = FeatureExtractionParameters(hop_length=1024, n_mfcc=25)
    metadata.calculate_all_features(fe_params) 

    def run_folder(folder):
        #get metadata train set excluding first directory
        train_set = metadata.train_set(folder)
        #get metadata test set for the selected folder
        test_set = metadata.test_set(folder)

        #extract features for each sound that belong to class c
        def forEachClass(c: SoundClass): 
            return np.array(list(map(lambda s: s.feature_extraction(fe_params), tqdm(c.positive)))) 

        X_train = np.array(list(map(forEachClass, train_set)))    #extract the training set features
        print(list(map(lambda x: len(x),X_train)))
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

        model = skl.svm.SVC(C=1, kernel='rbf')     #define the model
        model.fit(X_train, y_train)                 #train the model with the training set
        predictions = model.predict(X_test)         #make predictions on test set

        acc = skl.metrics.accuracy_score(y_test, predictions)      #compute the accuracy
        accuracies.append({"acc": acc, "folder": folder})
        folder_accuracy.append(acc)
        print(f'\nAccuracy: {acc}')

        predictions_all.extend(predictions)
        y_all.extend(y_test)

        skplt.metrics.plot_confusion_matrix(y_test, predictions, x_tick_rotation = 90, text_fontsize = 'small', figsize = (10,10))   #compute the confusion matrix
        plt.savefig('runs/'+run_id+'_folder_'+str(folder)+'.png')

        
    for folder in range(1, 11):
        run_folder(folder)

    acc_all = skl.metrics.accuracy_score(np.array(y_all), np.array(predictions_all)) 
    skplt.metrics.plot_confusion_matrix(np.array(y_all), np.array(predictions_all), x_tick_rotation = 90, text_fontsize = 'small', figsize = (10,10))   #compute the confusion matrix
    plt.savefig('runs/'+run_id+'_all.png')
    with open('runs/'+run_id + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Folder','Accuracy'])
        writer.writerow(['Average',"{:.2f}".format(acc_all)])
        writer.writerow(['Average',"{:.2f}".format(mean(folder_accuracy))])
        for acc in accuracies:
            writer.writerow([acc['folder'],acc['acc']])


if __name__ == "__main__":
    freeze_support()   # required to use multiprocessing
    main()
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import sklearn.svm
import IPython.display as ipd
import scipy as sp

def compute_mfcc(audio, fs, n_mfcc):
    X = np.abs(librosa.stft(
        audio, 
        window = 'hamming',
        n_fft = 1024,
        hop_length = 512,)
    )

    mel = librosa.filters.mel(
        sr=fs,
        n_fft = 1024,
        n_mels = 40,
        fmin=133.33,
        fmax=6853.8
    )

    melspectrogram = np.dot(mel, X)

    log_melspectrogram = np.log10(melspectrogram + 1e-16)
    mfcc = sp.fftpack.dct(log_melspectrogram, axis=0, norm='ortho')[1:n_mfcc+1]
    
    return mfcc

classes = ['blues', 'disco', 'metal']
n_mfcc = 13
dict_train_features = {'blues': [], 'metal': [], 'disco': []}

for c in classes:
  train_root = 'Input/{}/training/'.format(c)

  class_train_files = [f for f in os.listdir(train_root) if f.endswith('.wav')]
  n_train_samples = len(class_train_files)

  train_features = np.zeros((n_train_samples, n_mfcc))

  for index, f in enumerate(class_train_files):
    audio, fs = librosa.load(os.path.join(train_root, f), sr=None)
    mfcc = compute_mfcc(audio, fs, n_mfcc)
    train_features[index, :] = np.mean(mfcc, axis=1)

  dict_train_features[c] = train_features
    

dict_test_features = {'blues': [], 'metal': [], 'disco': []}

for c in classes:
    test_root = 'Input/{}/test/'.format(c)
    class_test_files = [f for f in os.listdir(test_root) if f.endswith('.wav')]
    n_test_samples = len(class_test_files)
    
    test_features = np.zeros((n_test_samples, n_mfcc))
    for index, f in enumerate(class_test_files):
        audio, fs = librosa.load(os.path.join(test_root, f), sr=None)
        mfcc = compute_mfcc(audio, fs, n_mfcc)
        test_features[index, :] = np.mean(mfcc, axis=1)
    dict_test_features[c] = test_features

for c in classes:
    mfcc = dict_train_features[c].transpose()
    # Visualization
    fig = plt.figure(figsize=(16, 6))
    plt.subplot(1,2,1)
    plt.imshow(mfcc, origin='lower', aspect='auto')
    plt.xlabel('Training samples')
    plt.ylabel('MFCC coefficients')
    plt.title('MFCC (coefficients 0 to 13) for class {}'.format(c))
    plt.colorbar()
    plt.tight_layout()

    mfcc_upper = mfcc[4:]
    plt.subplot(1,2,2)
    plt.imshow(mfcc_upper, origin='lower', aspect='auto')
    plt.title('MFCC (coefficients 4 to 13) for class {}'.format(c))
    plt.xlabel('Training samples')
    plt.ylabel('MFCC coefficients')
    plt.colorbar()
    plt.tight_layout()


class_0 = 'metal'
class_1 = 'disco'

X_train_0 = dict_train_features[class_0]
X_train_1 = dict_train_features[class_1]

X_train = np.concatenate((X_train_0, X_train_1), axis=0)

y_train_0 = np.zeros((X_train_0.shape[0]))
y_train_1 = np.ones((X_train_1.shape[0]))

y_train = np.concatenate((y_train_0, y_train_1), axis=0)

X_test_0 = dict_test_features[class_0]
X_test_1 = dict_test_features[class_1]

X_test = np.concatenate((X_test_0, X_test_1), axis=0)

y_test_0 = np.zeros((X_test_0.shape[0]))
y_test_1 = np.ones((X_test_1.shape[0]))

y_test = np.concatenate((y_test_0, y_test_1), axis=0)

feat_max = np.max(X_train, axis=0)
feat_min = np.min(X_train, axis=0)

X_train_normalized =  (X_train - feat_min) / (feat_max - feat_min)
X_test_normalized = (X_test - feat_min) / (feat_max - feat_min)

clf = sklearn.svm.SVC(C=1, kernel='rbf')

clf.fit(X_train_normalized, y_train)

def compute_metrics(gt_labels, predicted_labels):
    TP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 1))
    FP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 0))
    TN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 0))
    FN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 1))
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    print("Results : \n accuracy = {} \n precision = {} \n recall = {} \n F1 score = {}".format(
        accuracy, precision, recall, F1_score))


y_predict = clf.predict(X_test_normalized)

compute_metrics(y_test, y_predict)
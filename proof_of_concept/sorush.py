import os
import wave
import numpy
import csv
import scipy
import librosa
import yaml
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

catt = ('cat.wav')
dogg = ('dog.wav')
un = ('un.wav')
yc, fs =librosa.load(catt, sr=None)

yd, fs =librosa.load(dogg, sr=None)
yn, fs =librosa.load(un, sr=None)

def FE(input):

 window = scipy.signal.hamming(round(44100*0.04), sym=False)
 include_mfcc0: False
 include_delta: True
 include_acceleration: True
 power_spectrogram = numpy.abs(librosa.core.stft(input,
                                               n_fft=2048,
                                               win_length=round(44100*0.04),
                                               hop_length=round(44100*0.02),
                                               center=True,
                                               window=window))**2
 mel_basis = librosa.filters.mel(sr=fs,
                                    n_fft=2048,
                                    n_mels=40,
                                    fmin=0,
                                    fmax=22050,
                                    htk=False)
 mel_spectrum = numpy.dot(mel_basis, power_spectrogram)
 mfcc = librosa.feature.mfcc(S=librosa.amplitude_to_db(mel_spectrum),n_mfcc=20)
 feature_matrix = mfcc
 # Delta coefficients
 mfcc_delta = librosa.feature.delta(mfcc)
 # Add Delta Coefficients to feature matrix
 feature_matrix = numpy.vstack((feature_matrix))
 # Acceleration coefficients (aka delta delta)
 mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
 # Add Acceleration Coefficients to feature matrix
 feature_matrix = numpy.vstack((feature_matrix, mfcc_delta2))
 # Omit mfcc0
 feature_matrix = feature_matrix[1:, :]
 feature_matrix = feature_matrix.T
 return feature_matrix

C_Features=FE(yc)
D_Features=FE(yd)
u_Features=FE(yn)


dog = { "positive": {},
        "negetive": {}
}

cat = { "positive": {},
        "negetive": {}
}

models ={ "cat" : cat,
          "dog" : dog
}
model_container={ "models" :models
                }
                

model_container['models']['dog']['positive'] = BayesianGaussianMixture(n_components= 16,          
                                                       covariance_type='diag',      
                                                       random_state= 0,
                                                       tol= 0.001,
                                                       reg_covar= 0.001,
                                                       max_iter= 40,
                                                       n_init= 1
                                                       )
model_container['models']['dog']['positive'].fit(D_Features)

model_container['models']['dog']['negative'] = BayesianGaussianMixture(n_components= 16,          
                                                       covariance_type='diag',      
                                                       random_state= 0,
                                                       tol= 0.001,
                                                       reg_covar= 0.001,
                                                       max_iter= 40,
                                                       n_init= 1
                                                       )
model_container['models']['dog']['negative'].fit(C_Features)                                                       

BayesianGaussianMixture(covariance_prior=None, covariance_type='diag',
                        degrees_of_freedom_prior=None, init_params='kmeans',
                        max_iter=40, mean_precision_prior=None, mean_prior=None,
                        n_components=16, n_init=1, random_state=0,
                        reg_covar=0.001, tol=0.001, verbose=0,
                        verbose_interval=10, warm_start=False,
                        weight_concentration_prior=None,
                        weight_concentration_prior_type='dirichlet_process')

model_container['models']['cat']['positive'] = BayesianGaussianMixture(n_components= 16,          
                                                       covariance_type='diag',      
                                                       random_state= 0,
                                                       tol= 0.001,
                                                       reg_covar= 0.001,
                                                       max_iter= 40,
                                                       n_init= 1
                                                       )
model_container['models']['cat']['positive'].fit(C_Features)                                                       

model_container['models']['cat']['negative']= BayesianGaussianMixture(n_components= 16,          
                                                       covariance_type='diag',      
                                                       random_state= 0,
                                                       tol= 0.001,
                                                       reg_covar= 0.001,
                                                       max_iter= 40,
                                                       n_init= 1
                                                       )
model_container['models']['cat']['negative'].fit(D_Features) 

BayesianGaussianMixture(covariance_prior=None, covariance_type='diag',
                        degrees_of_freedom_prior=None, init_params='kmeans',
                        max_iter=40, mean_precision_prior=None, mean_prior=None,
                        n_components=16, n_init=1, random_state=0,
                        reg_covar=0.001, tol=0.001, verbose=0,
                        verbose_interval=10, warm_start=False,
                        weight_concentration_prior=None,
                        weight_concentration_prior_type='dirichlet_process')


def event_detection(feature_data, model_container, hop_length_seconds=0.01, smoothing_window_length_seconds=1.0, decision_threshold=0.0, minimum_event_length=0.1, minimum_event_gap=0.1):

    smoothing_window = int(smoothing_window_length_seconds / hop_length_seconds)

    results = []
    for event_label in model_container['models']:
        positive = model_container['models'][event_label]['positive'].score_samples(feature_data)
        negative = model_container['models'][event_label]['negative'].score_samples(feature_data)

        # Lets keep the system causal and use look-back while smoothing (accumulating) likelihoods
        for stop_id in range(0, feature_data.shape[0]):
            start_id = stop_id - smoothing_window
            if start_id < 0:
                start_id = 0
            positive[start_id] = sum(positive[start_id:stop_id])
            negative[start_id] = sum(negative[start_id:stop_id])

        likelihood_ratio = positive - negative
        event_activity = likelihood_ratio > decision_threshold

        # Find contiguous segments and convert frame-ids into times
        event_segments = contiguous_regions(event_activity) * hop_length_seconds

        # Preprocess the event segments
        event_segments = postprocess_event_segments(event_segments=event_segments,
                                                   minimum_event_length=minimum_event_length,
                                                   minimum_event_gap=minimum_event_gap)

        for event in event_segments:
            results.append((event[0], event[1], event_label))

    return results


def contiguous_regions(activity_array):


    # Find the changes in the activity_array
    change_indices = numpy.diff(activity_array).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = numpy.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = numpy.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def postprocess_event_segments(event_segments, minimum_event_length=0.1, minimum_event_gap=0.1):
  

    # 1. remove short events
    event_results_1 = []
    for event in event_segments:
        if event[1]-event[0] >= minimum_event_length:
            event_results_1.append((event[0], event[1]))

    if len(event_results_1):
        # 2. remove small gaps between events
        event_results_2 = []

        # Load first event into event buffer
        buffered_event_onset = event_results_1[0][0]
        buffered_event_offset = event_results_1[0][1]
        for i in range(1, len(event_results_1)):
            if event_results_1[i][0] - buffered_event_offset > minimum_event_gap:
                # The gap between current event and the buffered is bigger than minimum event gap,
                # store event, and replace buffered event
                event_results_2.append((buffered_event_onset, buffered_event_offset))
                buffered_event_onset = event_results_1[i][0]
                buffered_event_offset = event_results_1[i][1]
            else:
                # The gap between current event and the buffered is smalle than minimum event gap,
                # extend the buffered event until the current offset
                buffered_event_offset = event_results_1[i][1]

        # Store last event from buffer
        event_results_2.append((buffered_event_onset, buffered_event_offset))

        return event_results_2
    else:
        return event_results_1

current_results = event_detection(feature_data=u_Features,model_container=model_container)
print(current_results)




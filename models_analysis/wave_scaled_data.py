from .time_series_model import TimeSeriesData
import pywt
from collections import defaultdict, Counter
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_featuresV(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    # features = {"n5":entropy[0], }
    features = [entropy] + crossings + statistics
    # features = pd.Series(features)
    dict_features = [{'entropy':features[0],
                     'n5':features[1],
                     'n25':features[2],
                     'n95':features[3],
                     'median':features[4],
                     'std':features[5],
                     'var':features[6],
                     'rms':features[7],
                     'crossing':features[9]}]
    df_features = pd.DataFrame(dict_features)
    return df_features

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    features = [entropy] + crossings + statistics
    return features

def get_uci_har_features(dataset, labels, waveletname):
    uci_har_features = []
    for signal_no in range(0, len(dataset)):
        features = []
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            list_coeff = pywt.wavedec(signal, waveletname)
            for coeff in list_coeff:
                print("coef ", coeff)
                features += get_features(coeff)
        uci_har_features.append(features)
    X = np.array(uci_har_features)
    Y = np.array(labels)
    return X, Y

def extract_wave_features(data):
    slope_features = []
    # data = np.array([np.array(lst) for lst in data])
    for signal_no in range(0, data.shape[0]):
        features = []
        for signal in data.columns.array:
            signal = data[signal][signal_no]
            # print("signal ", signal, type(signal))
            list_coeff = pywt.wavedec(signal, 'rbio3.1')
            # print(list_coeff)
            for coeff in list_coeff:
                # print("coef ", coeff)
                features+=get_features(coeff)
        slope_features.append(features)
    
    scaler = StandardScaler()
    X = np.array(slope_features)
    X_ss = scaler.fit_transform(X)
    
    return X_ss

def get_wave_scaled_data(dataset):
    slope = extract_wave_features(dataset[['Slope']])
    path_eff = extract_wave_features(dataset[['Path_Efficiency']])
    coords = extract_wave_features(dataset[['CartesianX','CartesianY']])
    coords_slope = extract_wave_features(dataset[['CartesianX','CartesianY','Slope']])
    
    time_series = [
               TimeSeriesData("Slope", slope),
               TimeSeriesData("Path_eff",path_eff),
               TimeSeriesData("Coods_slope",coords_slope),
               TimeSeriesData("Coords", coords)
              ]
    return time_series
    
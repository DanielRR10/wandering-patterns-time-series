import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .time_series_model import TimeSeriesData
SIZE_SAMPLE = 80

def concatenate_arrays(data):
    
    values_arrays = []

    for arr in data.values:
        values_arrays.append(arr)
        
    return np.concatenate(values_arrays, axis=None)

def create_scaler(data):
    values_arrays = []

    for arr in data.values:
        values_arrays.append(arr)
    
    x = np.concatenate(values_arrays, axis=None)

    min_max_scaler = MinMaxScaler()
    scaler = min_max_scaler.fit(x.reshape(-1, 1))
    return scaler


def get_raw_scaled_data(dataset):
    coordsx_scaler = create_scaler(dataset['CartesianX'])
    coordsy_scaler = create_scaler(dataset['CartesianY'])
    slope_scaler = create_scaler(dataset['Slope'])
    path_eff_scaler = create_scaler(dataset['Path_Efficiency'])
    
    dataset['CartesianX'] =  dataset['CartesianX'].apply(lambda x: coordsx_scaler.transform(x.reshape(-1, 1)).flatten())
    dataset['CartesianY'] =  dataset['CartesianY'].apply(lambda x: coordsy_scaler.transform(x.reshape(-1, 1)).flatten())
    dataset['Slope'] =  dataset['Slope'].apply(lambda x: slope_scaler.transform(x.reshape(-1, 1)).flatten())
    dataset['Path_Efficiency'] =  dataset['Path_Efficiency'].apply(lambda x: path_eff_scaler.transform(x.reshape(-1, 1)).flatten())


    cartesian_x = dataset['CartesianX'].to_numpy().tolist()
    cartersian_y = dataset['CartesianY'].to_numpy().tolist()
    slope = dataset['Slope'].to_numpy().tolist()
    path_eff = dataset['Path_Efficiency'].to_numpy().tolist()
    
    parsed_list = dataset['Coords_Slope'].apply(lambda x:np.asarray([np.asarray(y) for y in x]))
    coords_slope = np.stack(parsed_list,axis=1).reshape(1600,SIZE_SAMPLE*3)
    
    parsed_list = dataset['Coords'].apply(lambda x:np.asarray([np.asarray(y) for y in x]))
    coords = np.stack(parsed_list,axis=1).reshape(1600,SIZE_SAMPLE*2)

    time_series = [TimeSeriesData("Slope", slope),
               TimeSeriesData("Path_eff",path_eff),
               TimeSeriesData("Coods_slope",coords_slope),
               TimeSeriesData("Coords", coords)
              ]
    
    return time_series
    
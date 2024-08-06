import numpy as np

def min_max_normalization(data):
    data = np.asarray(data, dtype=np.float64)
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def z_score_normalization(data):
    data = np.asarray(data, dtype=np.float64)
    return (data - np.mean(data)) / np.std(data)

def ray_removal_threshold(data, threshold=1000):
    data = np.asarray(data, dtype=np.float64)
    return np.clip(data, a_min=None, a_max=threshold)

def baseline_correction(data, offset=0):
    data = np.asarray(data, dtype=np.float64)
    return data - np.min(data) + offset

FILTERS = {
    'Normalization': {
        'min_max_normalization': min_max_normalization,
        'z_score_normalization': z_score_normalization,
    },
    'Background Removal': {
        'ray_removal_threshold': ray_removal_threshold,
        'baseline_correction': baseline_correction,
    }
}

FILTER_CONFIG = {
    'ray_removal_threshold': {
        'threshold': {
            'type': 'number',
            'label': 'Threshold',
            'default': 1000,
            'min': 0,
            'max': 10000,
            'step': 1
        }
    },
    'baseline_correction': {
        'offset': {
            'type': 'number',
            'label': 'Offset',
            'default': 0,
            'min': -1000,
            'max': 1000,
            'step': 1
        }
    }
}
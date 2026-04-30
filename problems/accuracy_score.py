import numpy as np

def accuracy_score(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))
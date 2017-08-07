import numpy as np

# https://stackoverflow.com/a/21230438/4126114
# Testing:
#    running_view(np.array([1,2,3,4,5,6,7,8,9,10]),3,0)
#    running_view(np.array([[1,2],[3,4],[5,6],[7,8],[9,10]]),3,0)
def running_view(arr, window, axis=-1):
    """
    return a running view of length 'window' over 'axis'
    the returned array has an extra last dimension, which spans the window
    """
    shape = list(arr.shape)
    shape[axis] -= (window-1)
    assert(shape[axis]>0)
    return np.lib.index_tricks.as_strided(
        arr,
        shape + [window],
        arr.strides + (arr.strides[axis],))

def _load_data_strides(data, n_prev = 100):  
    """
    data should be pd.DataFrame()
    """
    docX = running_view(data, n_prev, 0)
    docX = np.array([y.T for y in docX])
    return docX

def train_test_split(df, test_size=0.1, look_back=100):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    #X_train, y_train = _load_data(df.iloc[0:ntrn], y.iloc[0:ntrn])
    #X_test, y_test = _load_data(df.iloc[ntrn:], y.iloc[ntrn:])
    # alternative to the for loop in the original load data
    # Note that both the original load data and the stride consume a lot of memory
    X_train = _load_data_strides(df[:ntrn,:], look_back)
    X_test = _load_data_strides(df[ntrn:,:], look_back)

    return (X_train), (X_test)

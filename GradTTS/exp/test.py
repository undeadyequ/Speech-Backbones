import numpy as np

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


y= np.array([1, 1, 1, np.NaN, np.NaN, 2, 2, np.NaN, 0])

#nans, x= nan_helper(y)
#y[nans]= np.interp(x(nans), x(~nans), y[~nans])


def interpolate_nan(array_like):
    array = array_like.copy()
    nans = np.isnan(array)
    def get_x(a):
        return a.nonzero()[0]
    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])
    return array


y2 = np.array([1, 1, 1, 2, 2, 0])

print(np.all(np.isnan(y2) == False))

print(interpolate_nan(y))


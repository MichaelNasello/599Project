import numpy as np

# given a 1D array, this function returns a 2D array of sub-arrays
# ie. sliding_window([1,2,3,4], 2, 1) == [[1,2],[2,3],[3,4]]
# if the array isn't evenly split, the end is discarded
def sliding_window(data: list, window_size: int, overlap:int = 0):
    assert type(overlap) is int and overlap >= 0
    assert type(window_size) is int and window_size > 0
    assert window_size >= overlap
    return np.lib.stride_tricks.sliding_window_view(data,window_size)[::window_size-overlap]


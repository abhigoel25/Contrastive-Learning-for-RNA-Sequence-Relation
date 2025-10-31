import numpy as np
import scipy.interpolate as si
import torch.nn as nn
import torch

# --- B-Spline Helper Code ---
def get_S(n_bases=10, spline_order=3, add_intercept=True):
    S = np.identity(n_bases)
    m2 = spline_order - 1
    for i in range(m2):
        S = np.diff(S, axis=0)
    S = np.dot(S.T, S)
    # Add intercept (unpenalized row/col)
    if add_intercept:
        zeros_row = np.zeros((1, S.shape[1]))
        S = np.vstack([zeros_row, S])  # add zero row on top

        zeros_col = np.zeros((S.shape[0], 1))
        S = np.hstack([zeros_col, S])  # add zero col on left

    return S.astype(np.float32)
    # return (S + S.T) / 2

def get_knots(start, end, n_bases=10, spline_order=3):
    x_range = end - start
    start, end = start - x_range * 0.001, end + x_range * 0.001
    m = spline_order - 1
    nk = n_bases - m
    dknots = (end - start) / (nk - 1)
    return np.linspace(start - dknots * (m + 1), end + dknots * (m + 1), num=nk + 2 * m + 2)

def get_X_spline(x, knots, n_bases=10, spline_order=3, add_intercept=True):
    # tck = [knots, np.zeros(n_bases), spline_order - 1]
    tck = [knots, np.zeros(n_bases), spline_order]
    X = np.zeros([len(x), n_bases])
    for i in range(n_bases):
        vec = np.zeros(n_bases)
        vec[i] = 1.0
        tck[1] = vec
        X[:, i] = si.splev(x, tck, der=0)
    if add_intercept:
        intercept_col = np.ones((len(x), 1), dtype=np.float32)
        X = np.hstack([intercept_col, X])  # shape: (len(x), n_bases + 1)

    return torch.from_numpy(X) 
    # return X.astype(np.float32)

class BSpline:
    def __init__(self, start=0, end=1, n_bases=10, spline_order=3):
        self.start, self.end, self.n_bases, self.spline_order = start, end, n_bases, spline_order
        self.knots = get_knots(start, end, n_bases, spline_order)
        self.S = get_S(n_bases, spline_order, add_intercept=False)

    def predict(self, x):
        return get_X_spline(x, self.knots, self.n_bases, self.spline_order, add_intercept=False)

# --- PyTorch SplineWeight1D Module ---
class SplineWeight1D(nn.Module):
    def __init__(self, input_steps, input_filters, n_bases=10, spline_degree=3,
                 share_splines=False, l2_smooth=0.0, l2=0.0, use_bias=False):
        super().__init__()
        self.l2_smooth, self.l2 = l2_smooth, l2
        n_spline_tracks = 1 if share_splines else input_filters
        self.bs = BSpline(0, input_steps - 1, n_bases=n_bases, spline_order=spline_degree)
        self.positions = np.arange(input_steps)
        X_spline = self.bs.predict(self.positions)
        self.register_buffer('X_spline_K', X_spline.float())
        self.kernel = nn.Parameter(torch.zeros(n_bases, n_spline_tracks))
        self.bias = nn.Parameter(torch.zeros(n_spline_tracks)) if use_bias else None

    def forward(self, x):
        spline_track = torch.matmul(self.X_spline_K, self.kernel)
        if self.bias is not None:
            spline_track = spline_track + self.bias
        return x * (spline_track + 1)

    def get_regularization_loss(self):
        loss = 0.0
        if self.l2 > 0:
            loss += self.l2 * torch.sum(self.kernel**2)
        if self.l2_smooth > 0:
            S_matrix = torch.from_numpy(self.bs.S).float().to(self.kernel.device)
            wT_S_w = torch.matmul(torch.matmul(self.kernel.T, S_matrix), self.kernel)
            loss += self.l2_smooth * torch.mean(torch.diag(wT_S_w))
        return loss
import numpy as np
from scipy.stats import linregress
from numba import njit, prange, float64, int64
from numba.types import unicode_type, boolean
from numba.experimental import jitclass
from talib import BBANDS
from scipy.optimize import shgo


def hurst(p, l):
    """
    Arguments:
        p: ndarray -- the price series to be tested
        l: list of integers or an integer -- lag(s) to test for mean reversion
    Returns:
        Hurst exponent
    """
    if isinstance(l, int):
        lags = [1, l]
    else:
        lags = l
    assert lags[-1] >= 2, "Lag in prices must be greater or equal 2"
    print(f"Price lags of {list(lags)[1:]} are included")
    lp = np.log(p)
    var = [np.var(lp[l:] - lp[:-l]) for l in lags]
    hr = linregress(np.log(lags), np.log(var))[0] / 2
    return hr
    

def hl(ts):
    deltas = np.diff(ts)
    LR = linregress(ts[:-1], deltas)
    if LR[3] <= .05:
        out = -np.log(2)/LR[0]
    else:
        out = np.nan
    return out

        
def gen_CV(gazp, train=1):
    gazp = gazp.sort_values("Date").reset_index()
    idx = (
        gazp.groupby([gazp.Date.dt.year, gazp.Date.dt.isocalendar().week])
        .apply(lambda df: df.index.tolist())
        .tolist()
    )
    out = [
        ([el for lst in idx[i - train : i] for el in lst], idx[i])
        for i in range(train, len(idx))
    ]
    return out
    
def gen_GBM(N = 100000, mu = 0.1, std = 0.1, s0=100, random_seed = 42):
    np.random.seed(random_seed)
    mu = mu / N
    sigma = std / np.sqrt(N)
    t = np.arange(N)
    W = np.random.randn(N) 
    W = np.cumsum(W)*np.sqrt(1)
    X = (mu-0.5*sigma**2)*t + sigma*W 
    return s0*np.exp(X)
    
    
@njit
def crossup(x, y, i):
    if x[i - 1] < y[i - 1] and x[i] > y[i]:
        return 1
    else:
        return 0


@njit
def crossdown(x, y, i):
    if x[i - 1] > y[i - 1] and x[i] < y[i]:
        return 1
    else:
        return 0


@njit 
def gt(x,y,i):
    if x[i] > y[i]:
        return 1
    else:
        return 0
    
@njit 
def lt(x,y,i):
    if x[i] < y[i]:
        return 1
    else:
        return 0

spec = [
    ('x',float64[:]),
    ('y',float64[:]),
    ('how',unicode_type),
    ('acc', boolean),
    ('_acc', int64),
    ('i', int64)
]   

@jitclass(spec)
class Signal:
    
    def __init__(self, x, y, how, acc):
        self.x = x
        self.y = y
        if how is None or how not in ["crossup", "crossdown", "lower", "higher"]:
            raise ValueError(
                'how must be one of "crossup", "crossdown", "lower", or "higher"'
            )
        else:
            self.how = how
        self.acc = acc
        self._acc = 0
    
    def sig(self, i):
        
        if self.how == "crossup":
            out = crossup(self.x, self.y, i)
                
        elif self.how == "crossdown":
            out = crossdown(self.x, self.y, i)
            
        elif self.how == "higher":
            out = gt(self.x, self.y, i)

        elif self.how == "lower":
            out = lt(self.x, self.y, i)
            
        if self.acc:
            self._acc += out
            return self._acc
        else:
            return out
        

@njit
def BBANDS_return(price, upper, middle, lower, commission = .0003):
    L = price.shape[0]
    out = np.empty(L)
    out[:2] = 0.0
    out[-2:] = 0.0
    i = 1

    while i < (L-2):
        
        out[i+1] = 0.0
        
        if crossup(price, lower, i) and i < (L-2):
            out[i+1] = price[i+2]/(price[i+1]*(1+commission)) - 1.0
            trade = True
            while trade and i < (L-3):
                i += 1
                out[i+1] = price[i+2]/price[i+1] - 1.0
                if crossup(price, middle,i) or crossdown(price, lower, i):
                    trade = False
                    out[i+1] = (1-commission)*price[i+2]/price[i+1] - 1.0
        
        
        if crossdown(price, upper, i) and i < (L-2):
            out[i+1] = (1-commission)*price[i+1]/price[i+2] - 1.0
            trade = True
            while trade and i < (L-3):
                i += 1
                out[i+1] = price[i+1]/price[i+2] - 1.0
                if crossup(price, upper, i) or crossdown(price, middle, i):
                    trade = False
                    out[i+1] = price[i+1]/(price[i+2]*(1+commission)) - 1.0
         
        
        i +=1
    return out
    

@njit
def BBANDS_signal(price, upper, middle, lower):

    L = price.shape[0]
    out = np.empty(L, dtype=np.int64)
    out[0] = 0
    i = 1

    while i < L:
    
        out[i] = 0
                
        if crossup(price, lower, i):
            out[i] = 1
            trade = True
            while trade and i < (L - 1):
                i += 1
                out[i] = 1
                if crossup(price, middle, i) or crossdown(price, lower, i):
                    trade = False

        if crossdown(price, upper, i):
            out[i] = -1
            trade = True
            while trade and i < (L - 1):
                i += 1
                out[i] = -1
                if crossdown(price, middle, i) or crossup(price, upper, i):
                    trade = False
  
        i +=1
          
    return out
    
    
def gen_BBANDS(ts, idx, lb, ss, commission = 0.0003):
    '''
    Arguments:
        ts: np.ndarray - Close prices
        idx: [int]- CV id
        l: int - lookback
        ss: float - number of std above/below
    Returns:
        return generated by the BB strategy        
    '''
    min_idx = idx[0]
    assert min_idx > lb, "Not enough data given lookback!"
    max_idx = idx[-1]
    lb = int(lb)
    upper, middle, lower = BBANDS(ts[min_idx-lb:max_idx+1], lb, ss, ss)
    upper, middle, lower = upper[lb:], middle[lb:], lower[lb:]
    ts = ts[idx]
    return BBANDS_return(ts, upper, middle, lower, commission)
    
    
def optimize_BBANDS(ts, idx, l = (20,50), ss = (1.0,2.0), iters = 5, commission = .0003):
    def func(X, ts, idx, commission = commission):
        ret = gen_BBANDS(ts, idx, *X, commission)
        return -(np.multiply.reduce(1+ret) - 1)
    return shgo(func, (l,ss), args=(ts, idx, commission), iters=iters)
    

@njit(parallel=True)    
def SNR(s):
    '''
    SNR for an array
    '''
    ds = np.abs(s[-1] - s[0])
    L = s.shape[0]
    sum_ = np.empty(L-1)
    for i in prange(L-1):
        sum_[i] = np.abs(s[i+1] - s[i])
    return ds/np.sum(sum_)
    
@njit(parallel=True)
def get_SNR(s, idx, l=100):
    '''
    SNR for all datapoints in array, with lookback and warmup
    '''
    min_idx = idx[0]
    max_idx = idx[-1]
    L = max_idx - min_idx + 1
    out = np.empty(L)
    for i in prange(L):
        cursor = min_idx + i
        ds = np.abs(s[cursor] - s[cursor - l])
        acc = 0.0
        for k in range(cursor - l, cursor):  
            acc += np.abs(s[k+1] - s[k])  # endpoint inclusive
        out[i] = ds / acc
    return out

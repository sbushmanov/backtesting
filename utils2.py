from numba import njit, int64, float64
from numba.types import Array, string, boolean, Tuple, DictType
from numba.experimental import jitclass
import numpy as np


@njit
def crossup(x, y, i):
    """
    x,y: np.array
    i: int - point in time
    Returns: 1 or 0 when condition is met
    """
    if x[i - 1] < y[i - 1] and x[i] > y[i]:
        out = True
    else:
        out = False
    return out


@njit
def crossdown(x, y, i):
    if x[i - 1] > y[i - 1] and x[i] < y[i]:
        out = True
    else:
        out = False
    return out


sig_dt = Tuple([Tuple([Array(float64, 1, "C"), Array(float64, 1, "C"), string, boolean]), string])


spec = [("memory", boolean[:]), ("signals", DictType(int64, sig_dt)), ("L", int64)]


@jitclass(spec)
class MultiSig:
    def __init__(self, signals):
        L = len(signals)
        self.memory = np.array([False] * L)
        self.signals = {0: signals[0]}
        for i in range(1, L):
            self.signals[i] = signals[i]
        self.L = L

    def single_sig(self, signal, i, n):
        """
        Accepts:
            - signal: tuple of 4 fields
            - i: int - point in time
            - n: int - consequetive number of the signal
        Updates:
            - corresponding memory field
        Returns:
            - boolean accounting for memory
        """
        x, y, how, acc = signal
        if how == "crossup":
            out = crossup(x, y, i)
        elif how == "crossdown":
            out = crossdown(x, y, i)
        out = out | self.memory[n]
        if acc:
            self.memory[n] = out
        return out

    def sig(self, i):
        s, logic = self.signals[0]
        out = self.single_sig(s, i, 0)
        for cnt in range(1, self.L):
            s = self.single_sig(self.signals[cnt][0], i, cnt)
            out = out | s if logic == "or_" else out & s
            logic = self.signals[cnt][1]
        return out

    def reset(self):
        self.memory = np.array([False] * self.L)

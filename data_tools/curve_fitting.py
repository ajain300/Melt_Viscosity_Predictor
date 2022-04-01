from scipy.optimize import curve_fit
import numpy as np

# def Mw_func_obj(M, a1, a2, k1 , k2, Mcr):
#     y = []
#     for m in M:
#         if m > Mcr:
#             a = a2
#             k = k2
#         else:
#             a = a1
#             k = k1
#         y.append(k + a*m)
#     return np.array(y)

def Mw_func_obj(x, a1, a2, Mcr, Mcr_y):
    y = np.piecewise(x, [x < Mcr, x >= Mcr],
        [lambda x:a1*x + Mcr_y-(a1*Mcr), lambda x: a2*x + Mcr_y-(a2*Mcr)])
    return y

def fit_Mw(M, Y):
    vars_m, _ = curve_fit(Mw_func_obj, M, Y, p0 = [1, 3.4,3.2, 1])
    a1, a2, Mcr, Mcr_y = vars_m
    k1 = Mcr_y-(a1*Mcr)
    k2 = Mcr_y-(a2*Mcr)
    return a1, a2, k1, k2, Mcr
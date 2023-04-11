from scipy.optimize import curve_fit
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import r2_score

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
    vars_m, _ = curve_fit(Mw_func_obj, M, Y, p0 = [1, 3.4,4, 1], bounds = ((-2, 0, 0, -2),(3, 6, 6, 10)))
    a1, a2, Mcr, Mcr_y = vars_m
    k1 = Mcr_y-(a1*Mcr)
    k2 = Mcr_y-(a2*Mcr)
    r2 = r2_score(Y, Mw_func_obj(M, *vars_m))
    return a1, a2, k1, k2, Mcr, r2


################
def Mw_func_obj_2(x, a2, Mcr, k2):
    y = ((k2)*((Mcr)**(a2 - 1)))*(x) + ((k2)*x**a2)
    return y

def fit_Mw_2(M, Y):
    vars, _ = curve_fit(Mw_func_obj_2, M, Y, p0 = [1, 10**4,10**(-7)], maxfev = 5000) 
    return vars
#################
#################
def Mw_func_obj_3(x, a1,a2,Mcr,k2):
    y = np.piecewise(x, [x < Mcr, x >= Mcr],
        [lambda x:k2 + (a2-a1)*Mcr + a1*x, lambda x: k2 + a2*x])
    return y

def fit_Mw_3(M, Y):
    vars, _ = curve_fit(Mw_func_obj_3, M, Y, p0 = [1, 3.4, 4, -7])
    return vars
#################

def softplus(x, a1, b_1, k1,b_2, Mcr):
    y = (a1*x + k1) + b_1*np.log(1+ np.exp((b_2)*(x - Mcr)))
    return y

def fit_softplus(M, Y):
    vars, _ = curve_fit(softplus, M, Y, p0 = [1, 3.4, 1,1,0.5], maxfev = 7000, bounds = ((0, 0, -1, 0, 0),(2, 5, 1, 30, 1)))
    return vars

##################

def softplus_2(x, a1, b_1, k1,b_2, Mcr):
    y = (a1*(x-Mcr) + k1) + b_1*(np.log(1+ np.exp(b_2*(x - Mcr))) + x - Mcr)
    return y

def fit_softplus_2(M, Y):
    vars, _ = curve_fit(softplus_2, M, Y, p0 = [1, 3.4, 1,1,0.5], maxfev = 7000, bounds = ((0, 0, -1, 0, 0),(2, 5, 1, 30, 1)))
    return vars

##################

def softplus_3(x, a1, a2, k1,b, Mcr):
    y = (a1*(x-Mcr) + k1) + (1/b)*(a2-a1)*(np.log(1+ np.exp(-b*(x - Mcr))) + b*(x - Mcr))
    return y

def fit_softplus_3(M, Y):
    vars, _ = curve_fit(softplus_3, M, Y, p0 = [1, 3.4, 1,1,0.5], maxfev = 7000, bounds = ((0, 0, -1, 0, 0),(2, 5, 1, 30, 1)))
    return vars


def fit_Mw_manual(M, Y):
    pass

def shear_func_obj(S, Z_S, n, S_cr):
    V = np.log10(Z_S) - (1-n)*np.log10(1+(S/S_cr))
    return V

def fit_shear(S, Y):
    vars_s, _ = curve_fit(shear_func_obj, S, Y, p0 = [1E4, 0.7, 1E1])
    r2 = r2_score(Y, shear_func_obj(S, *vars_s))
    return *vars_s, r2

def shear_func_inf_obj(S, Z_S, n, S_cr, inf_S):
    V = np.log10(inf_S + ((Z_S-inf_S)/(1+(S/S_cr)**(1-n))))
    return V

def fit_shear_inf(S, Y):
    vars_s, _ = curve_fit(shear_func_obj, S, Y, p0 = [1E5, 0.8, 1E2, 1E2])
    return vars_s

def WLF_obj(T, Tr, C1, C2, eta_r):
    n = (C2 + (T-Tr))
    shift = (-1*C1*(T-Tr))/n
    out = eta_r + shift
    #print('obj func out')
    #print(Tr, C1, C2, eta_r)
    # print(out)
    return out

def fit_WLF(T, eta, sigma = None):
    if T[0] < 200:
        pT = T[0]
    else: 
        pT = 200

    vars_wlf, _ = curve_fit(WLF_obj, T, eta, p0 = [pT, 17.0,52.0, 8.0], bounds = ((160, -50, -50, -2),(T[0], 100, 500, 20)) ,sigma=sigma, maxfev = 7000)
    r2 = r2_score(eta, WLF_obj(T, *vars_wlf))
    return *vars_wlf, r2
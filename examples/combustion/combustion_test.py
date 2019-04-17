import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import tensorly
import matplotlib.ticker as ticker
import tensorsketch
from tensorsketch.tensor_approx import TensorApprox
import warnings
warnings.filterwarnings('ignore')
import scipy.io
from tensorsketch.sketch import Sketch



def simrun_name(name, inv_factor,rm_typ): 
    ''' 
    Create an file name for a simulation run
    '''
    return "data/"+name+"_frk"+str(inv_factor)+"_"+rm_typ+".pickle"
def simplot_name(name, inv_factor): 
    return "plots/"+name+"_frk"+str(inv_factor)+".pdf" 
    '''
    Create an file name for simulation result plots 
    ''' 

if __name__ == '__main__':
    #CO = pickle.load(open("data/CO.pickle", 'rb'))
    T = pickle.load(open("data/T.pickle", 'rb'))
    #P = pickle.load(open("data/P.pickle", 'rb'))
    #T = np.random.normal(size = (30,30,30) )
    inv_factor = 10
    ranks = (np.array(T.shape)/inv_factor).astype(int)    
    ks = (np.array(T.shape)/inv_factor).astype(int)*2
    ss = 2*ks+1 
    rm_typ = 'gprod'

    sim = tensorsketch.tensor_approx.TensorApprox(T, ranks, ks, ss, rm_typ = rm_typ)
    hooi_result = sim.tensor_approx('hooi')
    twopass_result = sim.tensor_approx('twopass')
    onepass_result = sim.tensor_approx('onepass') 
    pickle.dump([hooi_result, twopass_result, onepass_result], open('data/experiment.pickle','wb') )
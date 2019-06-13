import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import tensorly
import matplotlib.ticker as ticker
import tensorsketch
from tensorsketch.tensor_approx import TensorApprox
import warnings


# In[8]:


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


# In[9]:


def run_realdata_frk(data, inv_factor, name, random_seed = 1, rm_typ = "g"): 
    '''
    Run one pass, two pass, and HOOI on the same real world low-rank tensor data with 
    fixed rank = 1/inv_factor and varying k. 
    :param data: The dataset.
    :param inv_factor: The inverse of the increment between ks. 
    :param name: file name that store the simulation result. 
    '''
    ranks = (np.array(data.shape)/inv_factor).astype(int)
    _, _, _, hooi_rerr, _ = tensorsketch.tensor_approx.TensorApprox(data, ranks, rm_typ = rm_typ).tensor_approx('hooi')
    hooi_result = np.repeat(hooi_rerr, len(np.arange(2/inv_factor,2/5, (2/5 -1/inv_factor)/10))).tolist()
    one_pass_result = []
    two_pass_result = []
    for factor in np.arange(2/inv_factor,2/5, (2/5 -1/inv_factor)/10):  
        ks = (np.array(data.shape)*factor).astype(int)
        ss = 2*ks+1 
        sim = tensorsketch.tensor_approx.TensorApprox(data, ranks, ks, ss, rm_typ = rm_typ)
        _, _, _, two_pass_rerr, _ = sim.tensor_approx('twopass')
        _, _, _, one_pass_rerr, _ = sim.tensor_approx('onepass')
        one_pass_result.append(one_pass_rerr)
        two_pass_result.append(two_pass_rerr)
    result = [hooi_result, two_pass_result, one_pass_result] 
    pickle.dump( result, open(simrun_name(name,inv_factor,rm_typ), "wb" ) )
    return result

def run_realdata_fk(data,name,random_seed = 1, rm_typ = "g"): 
    '''
    Run one pass, two pass, and HOOI on the same real world low-rank tensor data with 
    same k and varying r. 
    :param data: The dataset.
    :param name: file name that store the simulation result. 
    '''
    X = data
    kratio = 1/4
    r0ratio = 1/20
    rratios = np.arange((r0ratio),(1/5),(1/100))
    dim = np.array(X.shape)
    _, _, _, hooi_rerr, _ = tensorsketch.tensor_approx.TensorApprox(X, (dim*r0ratio).astype(int), rm_typ = rm_typ).tensor_approx('hooi') 
    hooi_result = np.repeat(hooi_rerr, len(rratios))
    two_pass_result = np.zeros(len(rratios))
    one_pass_result = np.zeros(len(rratios))
    for idx, rratio in enumerate(rratios):   
        sim = tensorsketch.tensor_approx.TensorApprox(X, (dim*rratio).astype(int), (dim*kratio).astype(int), (dim*kratio).astype(int)*2+1, rm_typ = rm_typ)
        _, _, _, two_pass_rerr, _ = sim.tensor_approx('twopass')
        _, _, _, one_pass_rerr, _ = sim.tensor_approx('onepass')
        one_pass_result[idx] = one_pass_rerr
        two_pass_result[idx] = two_pass_rerr
    sim_list = [two_pass_result,one_pass_result,hooi_result]
    pickle.dump( sim_list, open("data/"+name+"_"+rm_typ+"_fk.pickle", "wb" ) )
    return sim_list
    
if __name__ == '__main__':   
    '''
        import scipy.io
        combustion = scipy.io.loadmat('data/data_1000E-02.mat').get('data') 
        combustion  =combustion[np.arange(200),:,:,:]
        r = (20,12,12,5)
        k = (40,25,25,5)
        s = 2*np.asarray(k)+1
        s[3] = 5
        print(s)
        _, _, _, hooi_rerr, _ = tensorsketch.tensor_approx.TensorApprox(combustion,r,k,s,rm_typ = 'ssrft').tensor_approx('hooi') 
        _, _, _, twopass_rerr, _ = tensorsketch.tensor_approx.TensorApprox(combustion,r,k,s,rm_typ = 'ssrft').tensor_approx('twopass') 
        _, _, _, onepass_rerr, _ = tensorsketch.tensor_approx.TensorApprox(combustion,r,k,s,rm_typ = 'ssrft').tensor_approx('onepass') 
        print(hooi_rerr)
        print(twopass_rerr)
        print(onepass_rerr)
    '''
    '''
    run_realdata_fk(combustion, "combustion", rm_typ = "ssrft")
    run_realdata_fk(combustion, "combustion", rm_typ = "u")
    run_realdata_fk(combustion, "combustion", rm_typ = "gprod")
    run_realdata_fk(combustion, "combustion", rm_typ = "g")
    
    run_realdata_frk(combustion, 8, "combustion", rm_typ = "ssrft")
    run_realdata_frk(combustion, 10,"combustion", rm_typ = "ssrft")
    run_realdata_frk(combustion, 15,"combustion", rm_typ = "ssrft")
    run_realdata_frk(combustion, 8, "combustion", rm_typ = "u")
    run_realdata_frk(combustion, 10,"combustion", rm_typ = "u")
    run_realdata_frk(combustion, 15,"combustion", rm_typ = "u")
    run_realdata_frk(combustion, 8, "combustion", rm_typ = "gprod")
    run_realdata_frk(combustion, 10,"combustion", rm_typ = "gprod")
    run_realdata_frk(combustion, 15,"combustion", rm_typ = "gprod")
    run_realdata_frk(combustion, 8, "combustion", rm_typ = "g")
    run_realdata_frk(combustion, 10,"combustion", rm_typ = "g")
    run_realdata_frk(combustion, 15,"combustion", rm_typ = "g")
   ''' 
 
    ABSORB = nc.Dataset("data/b.e11.BRCP85C5CNBDRD.f09_g16.023.cam.h0.ABSORB.208101-210012.nc").variables['ABSORB'][:]
    ABSORB = ABSORB.filled(ABSORB.mean())
    SRFRAD = nc.Dataset("data/b.e11.B1850C5CN.f09_g16.005.cam.h0.SRFRAD.040001-049912.nc").variables['SRFRAD'][:]
    BURDENDUST = nc.Dataset("data/b.e11.B1850C5CN.f09_g16.005.cam.h0.BURDENDUST.040001-049912.nc").variables['BURDENDUST'][:] 
    AODABS = nc.Dataset("data/b.e11.B1850C5CN.f09_g16.005.cam.h0.AODABS.040001-049912.nc").variables['AODABS'][:] 
    AODABS = AODABS.filled(AODABS.mean())


    run_realdata_fk(AODABS, "AODABS", rm_typ = "sp0")
    run_realdata_frk(AODABS, 8, "AODABS", rm_typ = "sp0")
    run_realdata_frk(AODABS, 10, "AODABS", rm_typ = "sp0")
    run_realdata_frk(AODABS, 15, "AODABS", rm_typ = "sp0")

    run_realdata_fk(SRFRAD, "SRFRAD", rm_typ = "sp0")
    run_realdata_frk(SRFRAD, 8, "SRFRAD", rm_typ = "sp0")
    run_realdata_frk(SRFRAD, 10, "SRFRAD", rm_typ = "sp0")
    run_realdata_frk(SRFRAD, 15, "SRFRAD", rm_typ = "sp0")

    run_realdata_fk(BURDENDUST, "BURDENDUST", rm_typ = "sp0")
    run_realdata_frk(BURDENDUST, 8, "BURDENDUST", rm_typ = "sp0")
    run_realdata_frk(BURDENDUST, 10, "BURDENDUST", rm_typ = "sp0")
    run_realdata_frk(BURDENDUST, 15, "BURDENDUST", rm_typ = "sp0")

    run_realdata_fk(ABSORB, "ABSORB", rm_typ = "sp0")
    run_realdata_frk(ABSORB, 8, "ABSORB", rm_typ = "sp0")
    run_realdata_frk(ABSORB, 10, "ABSORB", rm_typ = "sp0")
    run_realdata_frk(ABSORB, 15, "ABSORB", rm_typ = "sp0")


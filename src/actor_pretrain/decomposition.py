import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # Ignore some annoying warnings

"""
References:
Original Code:
    https://github.com/FateMurphy/CEEMDAN_LSTM/blob/main/CEEMDAN_LSTM/data_preprocessor.py
EMD, Empirical Mode Decomposition: 
    PyEMD: https://github.com/laszukdawid/PyEMD
    Laszuk & Dawid, 2017, https://doi.org/10.5281/zenodo.5459184
    EMD: Huang et al., 1998, https://doi.org/10.1098/rspa.1998.0193.
    EEMD: Wu & Huang, 2009, https://doi.org/10.1142/S1793536909000047.
    CEEMDAN: Torres et al., 2011, https://doi.org/10.1109/ICASSP.2011.5947265.
VMD, Variational mode decomposition: 
    vmdpy: https://github.com/vrcarva/vmdpy, 
    Vinícius et al., 2020, https://doi.org/10.1016/j.bspc.2020.102073.
    method: Dragomiretskiy & Zosso, 2014, https://doi.org/10.1109/TSP.2013.2288675.
SampEn, Sample Entropy:
    sampen: https://github.com/bergantine/sampen
K-Means, scikit-learn(sklearn):
    scikit-learn: https://github.com/scikit-learn/scikit-learn
"""

# 1.Decompose
# ------------------------------------------------------
# 1.Main. Decompose series

def decom(series=None, decom_mode='ceemdan', **kwargs): 
    """
    Decompose time series adaptively and return results in pd.Dataframe by PyEMD.EMD/EEMD/CEEMDAN or vmdpy.VMD.
    Example: df_decom = cl.decom(series, decom_mode='ceemdan')
    Plot by pandas: df_decom.plot(title='Decomposition Results', subplots=True)
    Input and Parameters:
    ---------------------
    series     - the time series (1D) to be decomposed
    decom_mode - the decomposing methods eg. 'emd', 'eemd', 'ceemdan', 'vmd'
    **kwargs   - any parameters of PyEMD.EMD(), PyEMD.EEMD(), PyEMD.CEEMDAN(), vmdpy.VMD()
               - eg. trials for PyEMD.CEEMDAN(), change the number of inputting white noise 
    Output:
    ---------------------
    df_decom   - the decomposing results in pd.Dataframe
    """

    try: series = pd.Series(series)
    except: raise ValueError('Sorry! %s is not supported to decompose, please input pd.Series, or pd.DataFrame(1D), nd.array(1D)'%type(series))
    decom_mode = decom_mode.lower()

    if decom_mode in ['emd', 'eemd', 'ceemdan']:
        try: import PyEMD
        except ImportError: raise ImportError('Cannot import EMD-signal!, run: pip install EMD-signal!')
        if decom_mode == 'emd': decom = PyEMD.EMD(**kwargs)
        elif decom_mode == 'eemd': decom = PyEMD.EEMD(**kwargs)
        elif decom_mode == 'ceemdan': decom = PyEMD.CEEMDAN(**kwargs)
        decom_result = decom(series.values).T
        df_decom = pd.DataFrame(decom_result, columns=['imf'+str(i) for i in range(len(decom_result[0]))])
    elif decom_mode == 'vmd':
        df_decom, imfs_hat, omega = decom_vmd(series, **kwargs)
    else: raise ValueError('%s is not a supported decomposition method!'%(decom_mode))
   
    if isinstance(series, pd.Series): 
        df_decom.index = series.index # change index

    df_decom['target'] = series
    df_decom.name = 'decom_mode_is_'+decom_mode.lower()
    return df_decom

# 1.1 VMD
def decom_vmd(series=None, alpha=2000, tau =0, K=10, DC=0, init=1, tol=1e-7, *kwargs):
    """
    Decompose time series by VMD, Variational mode decomposition by vmdpy.VMD.
    Example: df_vmd, imfs_hat, imfs_omega = cl.decom_vmd(series, alpha, tau, K, DC, init, tol)
    Plot by pandas: df_vmd.plot(title='VMD Decomposition Results', subplots=True)
    Input and Parameters:
    ---------------------
    series     - the time series (1D) to be decomposed
    alpha      - the balancing parameter of the data-fidelity constraint
    tau        - time-step of the dual ascent ( pick 0 for noise-slack )
    K          - the number of modes to be recovered
    DC         - true if the first mode is put and kept at DC (0-freq)
    init       - 0 = all omegas start at 0
                 1 = all omegas start uniformly distributed
                 2 = all omegas initialized randomly
    tol        - tolerance of convergence criterion eg. around 1e-6
    **kwargs   - any parameters of vmdpy.VMD()
    Output:
    ---------------------
    df_vmd     - the collection of decomposed modes in pd.Dataframe
    imfs_hat   - spectra of the modes
    imfs_omega - estimated mode center-frequencies    
    """
    try: import vmdpy
    except ImportError: raise ImportError('Cannot import vmdpy, run: pip install vmdpy')
    imfs_vmd, imfs_hat, imfs_omega = vmdpy.VMD(series, alpha, tau, K, DC, init, tol, **kwargs)
    df_vmd = pd.DataFrame(imfs_vmd.T, columns=['imf'+str(i) for i in range(K)])
    return df_vmd, imfs_hat, imfs_omega

# 2.Integrate
# ------------------------------------------------------
# 2.Main. Integrate

def inte(df_decom=None, inte_list=None, num_clusters=3):
    """
    Integrate IMFs to be CO-IMFs by sampen.sampen2 and sklearn.cluster.
    sampen.sampen2 및 sklearn.cluster를 통해 IMF를 통합하여 CO-IMF가 되도록 합니다.

    Example: df_inte = cl.inte(df_decom)
    Plot by pandas: df_inte.plot(title='Integrated IMFs (Co-IMFs) Results', subplots=True)
    Custom integration: please use cl.inte_sampen() and cl.inte_kmeans()
    Input and Parameters:
    ---------------------
    df_decom       - the decomposing results in pd.Dataframe, or a group of series
    inte_list      - the integration list, eg. pd.Dataframe, (int) 3, (str) '233', (list) [0,0,1,1,1,2,2,2], ...
    num_clusters   - number of categories/clusters eg. num_clusters = 3
    Output:
    ---------------------
    df_inte        - the integrating form of each time series
    """

    # check input
    try : df_decom = pd.DataFrame(df_decom)
    except: raise ValueError('Invalid input!')
    if 'target' in df_decom.columns:
        tmp_target = df_decom['target']
        df_decom = df_decom.drop('target', axis=1, inplace=False)
    else: tmp_target = None

    # check inte_list
    if inte_list is None: #without
        df_sampen = inte_sampen(df_decom)
        inte_list = inte_kmeans(df_sampen, num_clusters)
    
    ## inte_list is a pd.Dataframe type
    elif isinstance(inte_list, pd.DataFrame):
        if len(inte_list) == 1: inte_list == inte_list.T
    
    ## (str) '2', '23'
    elif type(inte_list) == str and len(inte_list) < df_decom.columns.size:
        df_list, n, c = {}, 0, 0
        for i in inte_list:
            for j in ['imf'+str(x) for x in range(n, n+int(i))]:
                df_list[j], n = c, n+1
            c += 1
        inte_list = pd.DataFrame(df_list, index=["Cluster"]).T
    
    ## (str) '233'
    elif type(inte_list) == str and len(inte_list) == df_decom.columns.size:
        inte_list = pd.DataFrame([int(x) for x in inte_list], columns=['Cluster'], index=['imf'+str(x) for x in range(len(inte_list))])    
    
    ## (int) 2
    elif type(inte_list) == int and inte_list < df_decom.columns.size:
        df_sampen = inte_sampen(df_decom)
        inte_list = inte_kmeans(df_sampen, inte_list)
    
    ## inte_list가 리스트 형식이나 np.numpy일때? (list) [0,0,1,1,1,2,2,2]
    else :
        try: inte_list = pd.DataFrame(inte_list, columns=['Cluster'], index=['imf'+str(x) for x in range(len(inte_list))])
        except: raise ValueError('Sorry! %s is an invalid integration list'%type(inte_list))


    # Integrate, name, and resort
    df_tmp = pd.DataFrame()
    for i in range(inte_list.values.max()+1):
        df_tmp['imf'+str(i)] = df_decom[inte_list[(inte_list['Cluster']==i)].index].sum(axis=1)

    df_inte = df_tmp.T # Use Sample Entropy sorting the Co-IMFs
    df_inte['sampen'] = inte_sampen(df_tmp).values
    df_inte.sort_values(by=['sampen'], ascending=False, inplace=True)
    df_inte.index = ['co-imf'+str(i) for i in range(inte_list.values.max()+1)]
    df_inte = df_inte.drop('sampen', axis=1, inplace=False).T    

    # Output
    df_inte.name = 'inte_list_is_'+''.join(str(x) for x in inte_list.values.ravel()) # record integrate list
    df_inte.index = df_decom.index
    if tmp_target is not None: df_inte['target'] = tmp_target # add tmp target column
    return df_inte


# 2.1 Sample Entropy
def inte_sampen(df_decom=None, max_len=1, tol=0.1, nor=True, **kwargs):
    """
    Calculate Sample Entropy for each IMF or series by sampen.sampen2.
    각 IMF 또는 시리즈에 대한 샘플 엔트로피를 sampen.sampen2로 계산합니다.

    Example: df_sampen = cl.inte_sampen(df_decom)
    Plot by pandas: df_sampen.plot(title='Sample Entropy')
    Input and Parameters:
    ---------------------
    df_decom   - the decomposing results in pd.Dataframe, or a group of series
    max_len    - maximum length of epoch (subseries)
    tol        - tolerance eg. 0.1 or 0.2
    nor        - normalize or not 
    **kwargs   - any parameters of sampen.sampen2()
    Output:
    ---------------------
    df_sampen  - the Sample Entropy of each time series in pd.Dataframe
    """
            
    if 'target' in df_decom.columns: df_decom = df_decom.drop('target', axis=1, inplace=False)
    try: import sampen
    except ImportError: raise ImportError('Cannot import sampen, run: pip install sampen!')
    np_sampen = []
    for i in range(df_decom.columns.size):
        sample_entropy = sampen.sampen2(list(df_decom['imf'+str(i)].values), mm=max_len, r=tol, normalize=nor, **kwargs)
        np_sampen.append(sample_entropy[1][1])
    df_sampen = pd.DataFrame(np_sampen, index=['imf'+str(i) for i in range(df_decom.columns.size)])
    return df_sampen


# 2.2 K-Means
def inte_kmeans(df_sampen=None, num_clusters=3, random_state=0, **kwargs):
    """
    Get integrating form by K-Means by sklearn.cluster.
    sklearn.cluster에서 K-Means로 integrating 양식을 가져옵니다.
    
    Example: inte_list = cl.inte_kmeans(df_sampen)
    Print: print(inte_list)
    Input and Parameters:
    ---------------------
    df_sampen    - the Sample Entropy of each time series in pd.Dataframe, or an one-column Dataframe with specific index
    num_clusters - number of categories/clusters eg. num_clusters = 3
    random_state - control the random state to guarantee the same result every time
    **kwargs     - any parameters of sklearn.cluster.KMeans()
    Output:
    ---------------------
    inte_list    - the integrating form of each time series in pd.Dataframe
    """

    # Get integrating form by K-Means
    try: 
        from sklearn.cluster import KMeans
    except ImportError: 
        raise ImportError('Cannot import sklearn, run: pip install sklearn!')
    
    np_inte_list = KMeans(n_clusters=num_clusters, random_state=random_state, **kwargs).fit_predict(df_sampen)
    inte_list = pd.DataFrame(np_inte_list, index=['imf'+str(i) for i in range(df_sampen.index.size)], columns=['Cluster'])
    return inte_list




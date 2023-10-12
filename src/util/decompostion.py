import numpy as np
from typing import  List
from PyEMD import EMD  #pip install EMD-signal


class Decomposition():
    @staticmethod
    def emd_data_transform1D(data, mode:str='emd') -> List[np.ndarray]:
        # mode에 따라서 데이터를 다르게hidden1 분해한다.
        # if mode == 'emd': decom = EMD()
        # elif mode == 'eemd' : decom = EEMD()
        # else : decom = CEEMDAN()
        if len(data.shape) == 2: data = np.squeeze(data, axis=-1)
        emd = EMD()
        IMFs = emd(data)
        data_res = data-IMFs[0]-IMFs[1]
        data_mf = np.stack((IMFs[0], IMFs[1], data_res), axis=-1)
        return data_mf  # 차원 확인

    @staticmethod
    def emd_data_transform2D(data, mode:str='emd') -> List[np.ndarray]:
        # mode에 따라서 데이터를 다르게 분해한다.
        # if mode == 'emd': decom = EMD()
        # elif mode == 'eemd' : decom = EEMD()
        # else : decom = CEEMDAN()
        # 2d data
        assert len(data.shape)==2 , "Your data dimension is %d. Use another function." % (data.ndim)
        time_steps, data_dims = data.shape #(40, 30)
        data_mf1 = []
        data_mf2 = []
        data_res = []
        for j in range(data_dims):
            S = np.ravel(data[:, j])
            emd = EMD()
            IMFs = emd(S)
            data_mf1.append(IMFs[0].tolist())
            data_mf2.append(IMFs[1].tolist())
            data_res.append((S-(IMFs[0]+IMFs[1])).tolist())
        data_mf1 = np.array(data_mf1).transpose([1, 0]) #(timeseries, asset)
        data_mf2 = np.array(data_mf2).transpose([1, 0]) #(timeseries, asset)
        data_res = np.array(data_res).transpose([1, 0]) 
        data_mf = np.stack((data_mf1, data_mf2, data_res), axis=-1)
        return data_mf  # (40,30,3)

    @staticmethod
    def emd_data_transform_per_batch(data, mode:str='emd') -> List[np.ndarray]:
        # mode에 따라서 데이터를 다르게 분해한다.
        # if mode == 'emd': decom = EMD()
        # elif mode == 'eemd' : decom = EEMD()
        # else : decom = CEEMDAN()
        
        # 3차원 데이터
        samples, time_steps, data_dims = data.shape
        data_mf1 = []
        data_mf2 = []
        for i in range(samples):
            sample_1 = []
            sample_2 = []
            for j in range(data_dims):
                S = np.ravel(data[i, :, j])
                emd = EMD()
                IMFs = emd(S)
                sample_1.append(IMFs[0].tolist())
                sample_2.append((S - IMFs[0]).tolist())
            data_mf1.append(sample_1)
            data_mf2.append(sample_2)
        data_mf1 = np.array(data_mf1).transpose([0, 2, 1]) #(batch, timeseries, asset)
        data_mf2 = np.array(data_mf2).transpose([0, 2, 1]) #(batch, timeseries, asset)
        data_mf = np.stack((data_mf1, data_mf2), axis=-1)
        return data_mf  # 차원 확인



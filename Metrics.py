import numpy as np

'''性能指标'''
class Metrics(object):
    def __init__(self,real_val,pre_val):
        self.real_val=real_val.astype(np.int64)
        self.pre_val=pre_val.astype(np.int64)
        self.ARE_val = -1
        self.AAE_val = -1
        self.F1_val = -1
        self.RE_val = -1
        self.F1_val_hh=-1
        self.real_distr=np.full((255,1),0).astype(np.int64)
        self.pre_distr=np.full((255,1),0).astype(np.int64)

    def ARE(self):
        relative_pre_error = self.pre_val / self.real_val - 1
        are=np.mean(np.abs(relative_pre_error))
        self.ARE_val=are
        return are

    def AAE(self):
        absolute_pre_error = self.pre_val - self.real_val
        aae=np.mean(np.abs(absolute_pre_error))
        self.AAE_val = aae
        return aae

    def F1(self):
        relative_pre_error = self.pre_val / self.real_val - 1
        recall=np.sum(relative_pre_error == 0) / self.real_val.shape[0]
        f1=2*recall/(1+recall)
        self.F1_val = f1
        return f1

    def F1_heavyhitter(self,top_nums):
        # real_top_ids=np.argsort(self.real_val)[-top_nums:]
        real_top_ids=np.where(self.real_val>top_nums)[0]
        # real_top_vals=self.real_val[real_top_ids]
        # pre_top_ids = np.argsort(self.pre_val)[-top_nums:]
        pre_top_ids = np.where(self.pre_val > top_nums)[0]
        # pre_top_vals = self.pre_val[pre_top_ids]
        common_ids = pre_top_ids[np.isin(pre_top_ids, real_top_ids)]
        if (pre_top_ids.shape[0]==0) | (real_top_ids.shape[0]==0):
            self.F1_val_hh = 0
            return 0
        pr=common_ids.shape[0]/pre_top_ids.shape[0]
        rr=common_ids.shape[0]/real_top_ids.shape[0]
        if pr+rr==0:
            self.F1_val_hh = 0
            return 0
        f1=2*pr*rr/(pr+rr)
        # f1=pr
        self.F1_val_hh = f1
        return f1

    def WMRE(self):
        real_unique_values, real_counts = np.unique(self.real_val[self.real_val<=self.real_distr.shape[0]], return_counts=True)
        self.real_distr[real_unique_values-1]=real_counts.reshape(-1,1)
        pre_vals=self.pre_val[self.pre_val >=1]
        pre_unique_values, pre_counts = np.unique(pre_vals[pre_vals<=self.real_distr.shape[0]],return_counts=True)
        self.pre_distr[pre_unique_values - 1] = pre_counts.reshape(-1,1)
        wmre=2*np.sum(np.abs(self.real_distr-self.pre_distr))/(np.sum(self.real_distr)+np.sum(self.pre_distr))
        self.WMRE_val=wmre
        return wmre

    def RE(self):
        i=np.array(range(len(self.real_distr))).astype(np.int64)
        i=i+1
        M_real = np.sum(self.real_distr * i)
        Entropy_real=-np.sum(self.real_distr * i * np.log(i/M_real)/M_real)
        M_pre = np.sum(self.pre_distr * i)
        Entropy_pre = -np.sum(self.pre_distr * i * np.log(i / M_pre) / M_pre )
        re=np.abs(Entropy_real-Entropy_pre)/np.abs(Entropy_real)
        self.RE_val=re
        return re

    def get_allval(self):
        self.ARE()
        self.AAE()
        self.F1()
        self.WMRE()
        self.RE()
        self.F1_heavyhitter(0.0002*np.sum(self.real_val))

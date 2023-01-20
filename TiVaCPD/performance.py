import numpy as np
from matplotlib import pyplot as plt
import warnings
from scipy.signal import find_peaks, peak_prominences
from sklearn import metrics
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import auc, roc_curve
import scipy
from torch import threshold
import pickle as pkl
from numpy import ma
from pyampd.ampd import find_peaks_adaptive
from itertools import product
import pandas as pd

class ComputeMetrics():
    def __init__(self, y_true:np.array, y_pred:np.array, moe:int=5, threshold=0.05, model_type = 'no', process=True):
        """
        @param y_true - ground truth binary labels for CPs
        @param y_pred - calculated binary labels for CPs
        @param moe - margin of error
        @param threshold - for filtering
        """
        super().__init__()
        self.y_true = y_true
        self.y_pred = y_pred
        self.moe = moe
        self.process = process
        self.threshold = threshold
        self.model_type=model_type

        self.f1, self.precision, self.recall, self.peaks= self.ChangePointF1Score(self.y_true, self.y_pred, window=moe, threshold=self.threshold)
        self.auc = self.get_auc(self.y_true, self.y_pred, self.model_type)


    def ChangePointF1Score_(self, y_true, y_pred, window, threshold):
        """Calculate the precision/recall of an estimated segmentation compared
        with the true segmentation.
        Args:
            y_true (list): array of the last index of each regime (true
                partition).
            y_pred (list): array of the last index of each regime (computed
                partition).
            window (int, optional): allowed error (in points).
        Returns:
            tuple: (precision, recall)
        """
        
        if self.process:

            if not np.all((y_pred == 0)):
                y_pred = self.post_processing(y_pred, threshold)
                y_pred[0:5] = 0 
                y_pred[len(y_pred)-5:len(y_pred)] = 0
                my_bkps = np.where(y_pred == 1)[0]
            else:
                my_bkps = []
        
        true_bkps = np.where(y_true == 1)[0]
        
        window = int(window)
        assert window > 0, "Window of error must be positive (window = {})".format(window)
        
        if len(true_bkps) == 0 and len(my_bkps) == 0:
            return 1,1, 1, y_pred

        if len(my_bkps) == 0:
            return 0, 0, 0, y_pred
        
        used = set()
        true_pos = set(
            true_b
            for true_b, my_b in product(true_bkps[:], my_bkps[:])
            if my_b - window < true_b < my_b + window
            and not (my_b in used or used.add(my_b))
        )

        tp_ = len(true_pos)
        
        precision = tp_ / (len(my_bkps) ) # TP/(TP+FP)
        
        try:
            recall = tp_ / (len(true_bkps) ) # TP/(TP+FN)
        except:
            import pdb; pdb.set_trace()

        if (precision + recall == 0): f1=0
        else:
            f1 = 2*precision*recall/(precision+recall)
        return f1, precision, recall, y_pred

    def ChangePointF1Score(self, y_true, y_pred, window, threshold):
        '''
        Input:
        gtLabel: binary array (0 when no CP observed, 1 when CP observed)
        label: binary array (0 when value < threshold, 1 when value > threshold aka CP observed)
        window: tolerated margin of error
    
        Output:
        f1, precision, recall scores
        '''
        if self.process:
            if not np.all((y_pred == 0)):
                y_pred = self.post_processing(y_pred, threshold)
                #first few and last few peaks are FPs so set those to 0
                y_pred[0:5] = 0 
                y_pred[len(y_pred)-5:len(y_pred)] = 0

        # given ground truth sequence of labels, and real labels, computes precision, recall and f1 score assuming leniancy of (window)
        l = len(y_true)
        window = int(window)
        # Compute precision
        tp=0
        
        totalLabel = np.sum(y_pred)

        estimatedCP = np.argwhere(y_pred==1).tolist()
        toKeep = estimatedCP.copy()
        for i in range(1,len(estimatedCP)):
            if estimatedCP[i][0] - estimatedCP[i-1][0] <= window:
                toKeep.remove([estimatedCP[i][0]])

        for i in toKeep:
            try:
                if (np.max(y_true[np.maximum(1,i[0]-window):np.minimum(l,i[0]+window)]) == 1): tp += 1
                
            except ValueError:
                pass
               
        fn = 0
        totalGt = np.sum(y_true)
        
        for i in np.argwhere(y_true==1):
            if (np.max(y_pred[np.maximum(0,i[0]-window):np.minimum(l-1,i[0]+window)]) == 1): fn += 1

        if totalLabel !=0:
            precision = tp / totalLabel
        else:
            precision = 0

        recall = fn / totalGt
        if (precision + recall == 0): f1=0
        else:
            f1 = 2*precision*recall/(precision+recall)
        if (totalLabel==0):
            precision=0
        if (totalGt==0):
            recall=0

        return f1, precision, recall, y_pred

    def peak_prominences_(self, distances):
        """
        Adapted calculation of prominence of peaks, based on the original scipy code
    
        Args:
            distances: dissimarity scores
        Returns:
            prominence scores
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            all_peak_prom = peak_prominences(distances, range(len(distances)))
        return all_peak_prom

    def post_processing(self, score, threshold):
        
        if pd.isnull(score).all():
            score = np.zeros(shape=score.shape)

        if not np.all((score==0)):
            
            score_peaks = np.zeros(shape = score.shape)
            peaks = find_peaks_adaptive(score)

            score_peaks[peaks] = 1
            
            return score_peaks 
        else:
            return score 



    def get_auc(self, y_true, y_pred, model_type):
        auc_scores = []
        minLength = int(min(len(y_true), len(y_pred)))
        if model_type=='KLCPD':
            y_pred = 2./(1+np.exp(-3*y_pred)) -1 # Logistic function for scaling bw 0-1
        
        if self.process:
            y_pred = self.post_processing(y_pred, threshold)
            y_pred[0:5] = 0 
            y_pred[len(y_pred)-5:len(y_pred)] = 0 
        
        fpr, tpr, _ = roc_curve(y_true[:minLength], y_pred[:minLength], pos_label = 1)
        auc_scores.append(auc(fpr, tpr))
        return auc_scores


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def save_data(path, array):
    with open(path,'wb') as f:
        pkl.dump(array, f)

def shift(arr, shift):
    r_arr = np.roll(arr, shift=shift)
    m_arr = ma.masked_array(r_arr,dtype=float)
    if shift > 0: m_arr[:shift] = ma.masked
    else: m_arr[shift:] = ma.masked
    return m_arr.filled(0)
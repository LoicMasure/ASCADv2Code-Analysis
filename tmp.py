import numpy as np
import scipy.stats as stats
import copy
import sys
import matplotlib.pyplot as plt
import ipdb

#AES Sbox
Sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

#permutation function for shuffling
G = np.array([0x0C, 0x05, 0x06, 0x0b, 0x09, 0x00, 0x0a, 0x0d, 0x03, 0x0e, 0x0f, 0x08, 0x04, 0x07, 0x01, 0x02])

def permIndices(i,m0,m1,m2,m3):
	x0,x1,x2,x3 = m0&0x0f, m1&0x0f, m2&0x0f, m3&0x0f
	return G[G[G[G[(15-i)^x0]^x1]^x2]^x3]

# Two Tables to process a field multplication over GF(256): a*b = alog (log(a) + log(b) mod 255)
log_table=[ 0,   0,  25,   1,  50,   2,  26, 198,  75, 199,  27, 104,  51, 238, 223,   3, 
	 100,   4, 224,  14,  52, 141, 129, 239,  76, 113,   8, 200, 248, 105,  28, 193,
	 125, 194,  29, 181, 249, 185,  39, 106,  77, 228, 166, 114, 154, 201,   9, 120 ,
	 101,  47, 138,   5,  33,  15, 225,  36,  18, 240, 130,  69,  53, 147, 218, 142 ,
	 150, 143, 219, 189,  54, 208, 206, 148,  19,  92, 210, 241,  64,  70, 131,  56 ,
	 102, 221, 253,  48, 191,   6, 139,  98, 179,  37, 226, 152,  34, 136, 145,  16 ,
	 126, 110,  72, 195, 163, 182,  30,  66,  58, 107,  40,  84, 250, 133,  61, 186 ,
	  43, 121,  10,  21, 155, 159,  94, 202,  78, 212, 172, 229, 243, 115, 167,  87 ,
	 175,  88, 168,  80, 244, 234, 214, 116,  79, 174, 233, 213, 231, 230, 173, 232 ,
	  44, 215, 117, 122, 235,  22,  11, 245,  89, 203,  95, 176, 156, 169,  81, 160 ,
	 127,  12, 246, 111,  23, 196,  73, 236, 216,  67,  31,  45, 164, 118, 123, 183 ,
	 204, 187,  62,  90, 251,  96, 177, 134,  59,  82, 161, 108, 170,  85,  41, 157 ,
	 151, 178, 135, 144,  97, 190, 220, 252, 188, 149, 207, 205,  55,  63,  91, 209 ,
	  83,  57, 132,  60,  65, 162, 109,  71,  20,  42, 158,  93,  86, 242, 211, 171 ,
	  68,  17, 146, 217,  35,  32,  46, 137, 180, 124, 184,  38, 119, 153, 227, 165 ,
	 103,  74, 237, 222, 197,  49, 254,  24,  13,  99, 140, 128, 192, 247, 112,   7 ]


alog_table =[1,   3,   5,  15,  17,  51,  85, 255,  26,  46, 114, 150, 161, 248,  19,  53 ,
	  95, 225,  56,  72, 216, 115, 149, 164, 247,   2,   6,  10,  30,  34, 102, 170 ,
	 229,  52,  92, 228,  55,  89, 235,  38, 106, 190, 217, 112, 144, 171, 230,  49 ,
	  83, 245,   4,  12,  20,  60,  68, 204,  79, 209, 104, 184, 211, 110, 178, 205 ,
	  76, 212, 103, 169, 224,  59,  77, 215,  98, 166, 241,   8,  24,  40, 120, 136 ,
	 131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206,  73, 219, 118, 154 ,
	 181, 196,  87, 249,  16,  48,  80, 240,  11,  29,  39, 105, 187, 214,  97, 163 ,
	 254,  25,  43, 125, 135, 146, 173, 236,  47, 113, 147, 174, 233,  32,  96, 160 ,
	 251,  22,  58,  78, 210, 109, 183, 194,  93, 231,  50,  86, 250,  21,  63,  65 ,
	 195,  94, 226,  61,  71, 201,  64, 192,  91, 237,  44, 116, 156, 191, 218, 117 ,
	 159, 186, 213, 100, 172, 239,  42, 126, 130, 157, 188, 223, 122, 142, 137, 128 ,
	 155, 182, 193,  88, 232,  35, 101, 175, 234,  37, 111, 177, 200,  67, 197,  84 ,
	 252,  31,  33,  99, 165, 244,   7,   9,  27,  45, 119, 153, 176, 203,  70, 202 ,
	  69, 207,  74, 222, 121, 139, 134, 145, 168, 227,  62,  66, 198,  81, 243,  14 ,
	  18,  54,  90, 238,  41, 123, 141, 140, 143, 138, 133, 148, 167, 242,  13,  23 ,
	  57,  75, 221, 124, 132, 151, 162, 253,  28,  36, 108, 180, 199,  82, 246,   1 ]

def multGF256(a,b):
    if (a==0) or (b==0):
        return 0
    else:
        return alog_table[(log_table[a]+log_table[b]) %255]

def hw(target):
    """
    Returns the Hamming Weight of the input target.
    """
    return bin(target).count("1")


def leakage_model(traces, z):
    unique = np.unique(z)
    groupby = [traces[z == i] for i in unique]
    mu_z = np.array([trace.mean(axis=0) for trace in groupby])
    return mu_z


class RunningMean():
    """
    A simple class that maintains the running mean of a random
    variable.
    """
    def __init__(self):
        self.m = 0
        self.n = 0
    
    def update(self, x_n):
        """
        Updates the running mean with a new observation x_n of the random
        variable X.
        """
        #if not (isinstance(x_n, np.ndarray) and x_n.dtype == np.float):
        x_n = np.array(x_n, dtype=np.float)
        
        self.m = (self.n * self.m + x_n) / (self.n + 1)
        self.n += 1
    
    def __call__(self):
        return self.m

class RunningVar():
    """
    A simple class that maintains the running variance of a random
    variable.
    """
    def __init__(self):
        self.m = RunningMean()
        self.m2 = 0
        self.n = 0
    
    def update(self, x_n):
        """
        Updates the running variance with a new observation x_n of the
        random variable X.
        """
        #if not (isinstance(x_n, np.ndarray) and x_n.dtype == np.float):
        x_n = np.array(x_n, dtype=np.float)
        
        self.m.update(x_n)
        self.m2 = (self.n * self.m2 + x_n ** 2) / (self.n + 1)
        self.n += 1
    
    def __call__(self):
        return self.m2 - self.m()**2

class RunningTtest():
    """
    A simple class that maintains the running computation of a
    T-test.
    """
    def __init__(self):
        self.mu = [RunningMean(), RunningMean()]
        self.var = [RunningVar(), RunningVar()]
        self.n = [0, 0]
    
    def update(self, x, label):
        """
        Updates the statistical terms of the t-test with a new observation x
        belonging to the class label.
        """
        self.mu[label].update(x)
        self.var[label].update(x)
        self.n[label] += 1
    
    def __call__(self):
        # Computation of the t-stat
        num_stat = (self.mu[0]() - self.mu[1]())
        denom_stat = np.sqrt(self.var[0]() / self.n[0] + self.var[1]() / self.n[1])
        t_stat = num_stat / denom_stat
        
        # Computation of the degrees of freedom
        #num_df = (self.var[0]() / self.n[0] + self.var[1]() / self.n[1])**2
        #denom_df = (self.var[0]() / self.n[0])**2 / (self.n[0] - 1) + (self.var[1]() / self.n[1])**2 / (self.n[1] - 1)
        #df = num_df / denom_df
        
        # Returns the p-value
        #p = 2 * stats.t.cdf(t_stat, df=df)
        return t_stat
    
class RunningSNR():
    """
    A simple class that maintains the running computation of the SNR.
    """
    def __init__(self, n_classes):
        self.mu_z = [RunningMean() for b in range(n_classes)]
        self.var_traces = RunningVar()
    
    def update(self, x_n, z):
        """
        Updates the running SNR with a new observation x_n belonging
        to the class of index z.
        """
        self.mu_z[z].update(x_n)
        self.var_traces.update(x_n)
    
    def __call__(self):
        mu_z_call = [mu_z() for mu_z in self.mu_z]
        return np.var(mu_z_call, axis=0) / self.var_traces()

    
class RunningCorr():
    """
    A simple class that maintains the running correlation coefficient
    between two random variables X and Y.
    """
    def __init__(self):
        # The cumulative moments of order one and two for X and Y
        self.mx, self.my, self.mxy, self.mx2, self.my2 = 0, 0, 0, 0, 0
        # The number of steps
        self.n = 0
    
    def update(self, x_n, y_n):
        """
        Updates the running correlation with new observations of X and Y.
        All the moments are updated.
        x_n: (D,)
        y_n: (256,)
        """
        x_n = np.array(x_n, dtype=np.float)
        y_n = np.array(y_n, dtype=np.float)
        #self.mx, self.my = RunningMean(), RunningMean()
        #self.varx, self.vary = RunningVar(), RunningVar()
        self.mx = (self.n * self.mx + x_n) / (self.n + 1)
        self.my = (self.n * self.my + y_n) / (self.n + 1)
        # self.mxy = (self.n * self.mxy + x_n * y_n) / (self.n + 1)
        self.mxy = (self.n * self.mxy + x_n[:, None] * y_n[None, :]) / (self.n + 1)
        self.mx2 = (self.n * self.mx2 + x_n ** 2) / (self.n + 1)
        self.my2 = (self.n * self.my2 + y_n ** 2) / (self.n + 1)
        self.n += 1
    
    def __call__(self):
        """
        Computes the running correlation provided the cumulative 
        moments currently updated.
        """
        cov =  self.mxy - self.mx[:, None] * self.my[None, :]
        std_x = np.sqrt(self.mx2 - self.mx ** 2)
        std_y = np.sqrt(self.my2 - self.my ** 2)
        return cov / (std_x[:, None] * std_y[None, :])


class CPA():
    """
    A simple class to run a CPA attack
    """
    def __init__(self, traces, plains, keys, alphas=None, betas=None):
        """
        Params:
            * traces: np.ndarray of shape (N_a, D) denoting the traces.
            * plains: np.ndarray of shape (N_a,) denoting the plaintexts.
            * keys: np.ndarray of shape (N_a,) denoting the target keys.
            * alphas: np.ndarray of shape (N_a,) denoting the multiplicative mask.
            * betas: np.ndarray of shape (N_a,) denoting the additive mask.
        """
        # The attack set
        self.traces = traces
        self.N_a, self.D = self.traces.shape
        self.leakage_model = hw
        
        # Prepares the hypothesis matrix
        leak_model_vec = np.vectorize(self.leakage_model)
        z = [Sbox[plains ^ keys ^ k] for k in range(1 << 8)]
        if alphas is not None:
            mult_vec = np.vectorize(multGF256)
            target = mult_vec(alphas, z)
        else:
            target = z
        if betas is not None:
            target ^= betas
        self.h = leak_model_vec(target)  # h.shape = (256, N_a)
        
    def __call__(self, max_N_a=50):
        """
        Runs the CPA over the whole traces
        """
        curr_N_a = min(max_N_a, self.N_a)
        running_corr = RunningCorr()
        scores = np.zeros((curr_N_a, 1<<8))
        for i in range(curr_N_a):
            sys.stdout.write("\rProcessing trace {}\t".format(i))
            running_corr.update(self.traces[i], self.h[:, i])
            scores[i, :] = np.abs(running_corr()).max(axis=0) # corr: (D, 256)
        return scores
    
    def compute_ge(self, n_est=10, max_N_a=50):
        """
        Computes the success rate and the guessing entropy estimated n_est times.
        """
        curr_N_a = min(max_N_a, self.N_a)
        ranks = np.zeros((n_est, curr_N_a))
        for u in range(n_est):
            sys.stdout.write("\rProcessing estimation {}\t".format(u))
            order = np.random.permutation(self.traces.shape[0])
            self.traces = self.traces[order]
            self.h = self.h[:, order]
            scores = self.__call__(max_N_a=max_N_a)
            sorted_key_hyp = np.flip(scores.argsort(axis=1), axis=1)
            right_key = 0
            ranks[u, :] = np.where(right_key == sorted_key_hyp)[1]
        
        # Computes the SCA metrics
        ge = ranks.mean(axis=0)
        succ_rate = []
        for i in range(ranks.shape[1]): 
            succ_rate.append(np.where(ranks[:, i] < 1)[0].shape[0] / ranks.shape[0])
        succ_rate = np.array(succ_rate)
        return ge, succ_rate

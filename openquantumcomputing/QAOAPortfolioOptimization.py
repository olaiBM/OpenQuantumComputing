from qiskit import *
import numpy as np
import math
import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/olaib/QuantumComputing/OpenQuantumComputing')

from openquantumcomputing.QAOAQUBO import QAOAQUBO

class QAOAPortfolioOptimization(QAOAQUBO):

    def __init__(self, params=None):
        super(QAOAPortfolioOptimization, self).__init__(params=params)

        self.__checkParams()
        self.risk = params.get("risk")
        self.budget = params.get("budget")
        self.cov_matrix = params.get("cov_matrix")
        self.exp_return = params.get("exp_return")
        self.penalty = params.get("penalty", 0.0)
        self.N_assets = len(self.exp_return)

        # Reformulated as a QUBO
        # min x^T Q x + c^T x + b
        # Writing Q as lower triangular matrix since it otherwise is symmetric
        Q = self.risk * np.tril(self.cov_matrix + np.tril(self.cov_matrix, k=-1)) \
                        + self.penalty*(np.eye(self.N_assets) + 2* np.tril(np.ones((self.N_assets, self.N_assets)), k=-1))
        c = - self.exp_return - (2*self.penalty*self.budget*np.ones_like(self.exp_return))
        b = self.penalty*self.budget*self.budget

        self._init_QUBO(Q=Q, c=c, b=b)


    def __checkParams(self):
        # we require the following params:
        for key in ["risk", "budget", "cov_matrix", "exp_return"]:
            assert(key in self.params), "missing required parameter " + key
         

    def __str2np(self, s):
        x = np.array(list(map(int, s)))
        assert(len(x) == len(self.params.get("exp_return"))), \
            "bitstring  " + s + " of wrong size. Expected " + str(len(self.params.get("exp_return"))) + " but got " + str(len(x))
        return x

    def cost_nonQUBO(self, string, penalize=True):
        
        risk       = self.params.get("risk")
        budget     = self.params.get("budget")
        cov_matrix = self.params.get("cov_matrix")
        exp_return = self.params.get("exp_return")
        penalty    = self.params.get("penalty", 0.0)

        x = np.array(list(map(int,string)))        
        cost = risk* (x.T@cov_matrix@x) - exp_return.T@x
        if penalize:
            tot_penalty = penalty * (x.sum() - budget)**2
            print("Total penalty: ", tot_penalty)
            cost += tot_penalty

        return -cost


    def isFeasible(self, string, feasibleOnly=False):
        x = self.__str2np(string)
        constraint = np.sum(x) - self.params.get("budget")
        if math.isclose(constraint, 0,abs_tol=1e-7):
            return True
        else:
            return False

  
from qiskit import *
import numpy as np
import math

import qiskit.quantum_info as qi
from qiskit.circuit.library import PauliEvolutionGate


from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
#run: pip install openquantumcomputing
import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/olaib/QuantumComputing/OpenQuantumComputing')
sys.path.append('/Users/olaib/QuantumComputing/OpenQuantumComputing_private')
from openquantumcomputing2.Mixer import *
from openquantumcomputing.QAOAQUBO import QAOAQUBO


class QAOAConstrainedQUBO(QAOAQUBO):

    def __init__(self, params=None):
        """
        init function that initializes QUBO.
        The aim is to solve the problem
        min x^T Q x + c^T x + b 
        for n-dimensional binary variable x

        :param params: additional parameters
        """
        super().__init__(params=params)

        # Related to constraint
        self.B = []
        self.best_mixer_terms = []
        self.mixer_circuit = None
        self.reduced = params.get("reduced", True)
        self.initial_state = None




    def create_mixer_circuit(self, d, q):  

        if not self.best_mixer_terms:
            self.computeBestMixer()
        #Does not depend on self.ruben??
        parameter_list = [None]*self.N_betas
        c = self.mixer_circuit.copy()
        for i in range(self.N_betas):
            parameter_list[i] = Parameter('beta_' + str(d) + str(i))
            c.assign_parameters({c.parameters[i]: parameter_list[i]}, inplace = True) 
        self.parameterized_circuit.compose(c, inplace = True)
        self.beta_params[d] = parameter_list

        usebarrier = self.params.get('usebarrier', False)
        if usebarrier:
            self.parameterized_circuit.barrier()



    

    def computeBestMixer(self):
        if not self.B:
            self.computeFeasibleSubspace()
        if not self.best_mixer_terms:
                
            m = Mixer(self.B, sort = True)
            m.compute_commuting_pairs()
            m.compute_family_of_graphs()
            m.get_best_mixer_commuting_graphs(reduced = self.reduced)
            self.mixer_circuit, self.best_mixer_terms, self.logical_X_operators= m.compute_parametrized_circuit(self.reduced)
        else:
            #has already computed mixer
            pass

    def computeFeasibleSubspace(self):
        """
        To be implemented by a child class,
        where a constraint is included and the feasbible subspace corresponding
        to this constraint is computed using this function
        """
        raise NotImplementedError
    

    def setToInitialState(self, quantum_register):
        #set to ground state of mixer hamilton??
        if not self.B:
            self.computeFeasibleSubspace()

        if self.ruben:
            self.initial_state = self.B[0]
            ampl_vec = np.zeros(2 ** len(self.initial_state))
            ampl = 1
            ampl_vec[int(self.B[0], 2)] = ampl
        else:
                

                    # initial state
            ampl_vec = np.zeros(2 ** len(self.B[0]))
            ampl = 1 / np.sqrt(len(self.B))
            for state in self.B:
                ampl_vec[int(state, 2)] = ampl
            
            
        self.parameterized_circuit.initialize(ampl_vec, quantum_register)

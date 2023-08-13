from qiskit import *
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import XXPlusYYGate
import numpy as np
import math
import itertools

#from openquantumcomputing.QAOAQUBO import QAOAQUBO

import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/olaib/QuantumComputing/OpenQuantumComputing')
from openquantumcomputing.QAOAConstrainedQUBO import QAOAConstrainedQUBO
from openquantumcomputing2.PauliString import PauliString

class QAOAKhot(QAOAConstrainedQUBO):

    def __init__(self, params=None):
        super().__init__(params=params)

        self.k = None #Number of ones in feasible strings. Must be initialized by a child class
        self.cascade = params.get("cascade", False)
        self.ring = params.get("ring", False)
 
    
    def __str2np(self, s):
        x = np.array(list(map(int, s)))
        assert(len(x) == self.N_qubits), \
            "bitstring  " + s + " of wrong size. Expected " + str(self.N_qubits) + " but got " + str(len(x))
        return x



    def isFeasible(self, string, feasibleOnly=False):
        x = self.__str2np(string)
        constraint = np.sum(x) - self.k
        if math.isclose(constraint, 0,abs_tol=1e-7):
            return True
        else:
            return False
        
    def computeBestMixer(self):
        #Overrides this function of QAOAConstrainedQUBO for the k-hot problem where structure of mixer is known
        if not self.best_mixer_terms:
            q = QuantumRegister(self.N_qubits) 
            c = ClassicalRegister(self.N_qubits)
            self.mixer_circuit = QuantumCircuit(q, c)
            self.best_mixer_terms, self.logical_X_operators = self.__XYMixerTerms()
            scale = 0.5 #Since every logical X has two stabilizers
            if not self.ruben:
                Beta = Parameter('Beta')
                 
                for i in range(self.N_qubits-1):
                    #Hard coded XY mixer
                    current_gate = XXPlusYYGate(scale*Beta)
                    self.mixer_circuit.append(current_gate, [i, i+1])
            else:
                self.N_gammas = 0
                if self.ring:
                    number = self.N_qubits
                        
                else:
                     number = self.N_qubits-1

                self.N_betas = number
                if self.cascade:

                    for i in range(self.N_qubits -1):
                        current_parameter = Parameter(f"xxx_{i}")     #Qiskit sorts parameters alphabetically using parameter names, must have suitable name
                        current_gate = XXPlusYYGate(scale*current_parameter)
                        self.mixer_circuit.append(current_gate, [i, i+1])

                    if self.ring:
                        current_parameter = Parameter(f"xxx_{self.N_qubits}")     #Qiskit sorts parameters alphabetically using parameter names, must have suitable name
                        current_gate = XXPlusYYGate(scale*current_parameter)
                        self.mixer_circuit.append(current_gate, [0, self.N_qubits -1])

                else:

                    last_1_index = self.k -1                            #index of last one in bit string used as initial state, big endian encoding
                    print("last one index, big endian: ", last_1_index)
                    last_1_index = (self.N_qubits -1) - last_1_index    #index of last one in bit string used as initial state, little endian encoding
                    print("last one index, little endian", last_1_index)
                    
                    index_counter = last_1_index
                    for i in range(number):
                        current_parameter = Parameter(f"xxx_{i}")     #Qiskit sorts parameters alphabetically using parameter names, must have suitable name
                        current_gate = XXPlusYYGate(scale*current_parameter)
                        if (index_counter -1 ) == -1:
                            self.mixer_circuit.append(current_gate, [0, self.N_qubits -1 ])
                            index_counter = self.N_qubits -1
                        else:
                            self.mixer_circuit.append(current_gate, [index_counter, index_counter-1])
                            index_counter = index_counter -1



                    



        
    def computeFeasibleSubspace(self):
        print("Its now computing the feasible subspace")
        for combination in itertools.combinations(range(self.N_qubits), self.budget):
            current_state = ['0']*self.N_qubits
            for index in combination:
                current_state[index] = '1'
            self.B.append(''.join(current_state))

    def __XYMixerTerms(self):
        #return ring XY mixer
    
        logical_X_operators = [None]*(self.N_qubits)
        mixer_terms = {}
        scale = 0.5                         #1/size, size of stabilizer space
        for i in range(self.N_qubits -2):
            logical_X_operator = ["I"]*(self.N_qubits-1)
            logical_X_operator[i] = "X"
            logical_X_operator[i+1] = "X"
            logical_X_operator = "".join(logical_X_operator)
            logical_X_operators[i] = logical_X_operator

            mixer_terms[logical_X_operator] = [PauliString(scale, logical_X_operator)]

            YY_operator = ["I"]*(self.N_qubits-1)
            YY_operator[i] = "Y"
            YY_operator[i+1] = "Y"
            YY_operator = "".join(YY_operator)

            mixer_terms[logical_X_operator].append(PauliString(scale, YY_operator))

        last_operator = ["I"]*(self.N_qubits-1)
        last_operator[0] = "X"
        last_operator[-1] = "X"
        last_operator = "".join(last_operator)

        last_operatorY = ["I"]*(self.N_qubits-1)
        last_operatorY[0] = "Y"
        last_operatorY[-1] = "Y"
        last_operatorY = "".join(last_operator)

        logical_X_operators[-1] =  last_operator
        mixer_terms[last_operator] = [PauliString(scale, last_operatorY)]

        return mixer_terms, logical_X_operators












            
        











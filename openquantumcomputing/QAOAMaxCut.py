from qiskit import *
from qiskit.circuit import Parameter
import numpy as np

from openquantumcomputing.QAOABase import QAOABase

class QAOAMaxCut(QAOABase):
    
    def __init__(self, params=None):
        super().__init__(params=params)

        self.G = self.params.get('G', None)
        self.N_qubits = self.G.number_of_nodes()



    def cost(self, string):

        C = 0
        for edge in self.G.edges():
            i = int(edge[0])
            j = int(edge[1])
            if string[i] != string[j]:
                w =self. G[edge[0]][edge[1]]['weight']
                C += w
        return C
    
    def create_cost_circuit(self, d, q):
        """
        Adds a parameterized circuit for the cost part to the member variable self.parameteried_circuit
        and a parameter to the parameter list self.gamma_params
        """
        self.gamma_params[d] = Parameter('gamma_' + str(d))
        usebarrier = self.params.get('usebarrier', False)
        if usebarrier:
            self.parameterized_circuit.barrier()

        ### cost Hamiltonian
        for edge in self.G.edges():
            i = int(edge[0])
            j = int(edge[1])
            w = self.G[edge[0]][edge[1]]['weight']
            wg = w * self.gamma_params[d]
            self.parameterized_circuit.cx(q[i], q[j])
            self.parameterized_circuit.rz(wg, q[j])
            self.parameterized_circuit.cx(q[i], q[j])
            if usebarrier:
                self.parameterized_circuit.barrier()
                

    def create_mixer_circuit(self, d, q):
       
        q = QuantumRegister(self.N_qubits) 
        c = ClassicalRegister(self.N_qubits)
        self.mixer_circuit = QuantumCircuit(q, c)
        if self.ruben:
            self.N_gammas = 0
            self.N_betas = self.N_qubits
            for i in range(self.N_qubits):
                current_parameter = Parameter(f"xxx_{i}")     #Qiskit sorts parameters alphabetically using parameter names, must have suitable name
                self.mixer_circuit.rx(current_parameter, i)
            
            parameter_list = [None]*self.N_betas
            c = self.mixer_circuit.copy()
            for i in range(self.N_betas):
                parameter_list[i] = Parameter('beta_' + str(d) + str(i))
                c.assign_parameters({c.parameters[i]: parameter_list[i]}, inplace = True) 
                print("c parameters: ", c.parameters)
            self.parameterized_circuit.compose(c, inplace = True)
            self.beta_params[d] = parameter_list


        else:

            self.beta_params[d] = Parameter('beta_'+str(d))
            self.mixer_circuit.rx(-2 * self.beta_params[d], range(self.N_qubits))

            self.parameterized_circuit.compose(self.mixer_circuit, inplace = True)

        usebarrier = self.params.get('usebarrier', False)
        if usebarrier:
            self.parameterized_circuit.barrier()

    def setToInitialState(self, q):
        """
        Implements the function that sets the member variable self.parameterized_circuit
        in the initial state
        :param q: The qubit register which is initialized

        """
        if self.ruben:
            init = ["0"]*self.N_qubits
            self.initial_state = "".join(init)
            ampl_vec = np.zeros(2 ** len(self.initial_state))
            ampl = 1
            ampl_vec[int(self.initial_state, 2)] = ampl
            self.parameterized_circuit.initialize(ampl_vec, q)

        else:
            self.parameterized_circuit.h(range(self.N_qubits))
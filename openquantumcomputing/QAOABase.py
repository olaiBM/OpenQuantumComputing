import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/olaib/QuantumComputing/OpenQuantumComputing')


from qiskit import *
import numpy as np
from scipy.optimize import minimize
import math, time


from openquantumcomputing.Statistic import Statistic

class QAOABase:

    def __init__(self,params = None):
        """
        init function that initializes member variables

        :param params: additional parameters
        """
        self.params=params
        self.E=None
        self.Var=None
        self.current_depth=0 # depth at which local optimization has been done
        self.angles_hist={} # initial and final angles during optimization per depth
        self.num_fval={} # number of function evaluations per depth
        self.t_per_fval={} # wall time per function evaluation per depth
        self.num_shots={} # number of total shots taken for local optimization per depth
        self.costval={} # optimal cost values per depth
        self.gamma_grid=None
        self.beta_grid=None
        self.stat=Statistic(alpha=self.params.get('alpha', 1))

        # Related to parameterized circuit
        self.use_parameterized_circuit = params.get("parameterize", False)
        self.parameterized_circuit = None
        self.current_circuit_depth = 0
        self.gamma_params = None
        self.beta_params = None
        self.ruben = params.get("ruben", False)
        self.hot_start = params.get("hotstart", False) #Good start for initial parameters?
        self.randomize_init = params.get("randomize_init", False)
        self.N_beta = None #Number of beta parameteres

        self.g_it=0
        self.g_values={}
        self.g_angles={}

        #self.parameterized = False

################################
# functions to be implemented:
################################

    def cost(self, string):
        """
        implements the cost function

        :param string: a binary string
        :return: a scalar value
        """
        raise NotImplementedError
    def cost_circuit(self, angles, depth):
        """
        Implements the function that returns the cost hamiltonian part of the circuit

        :param angles: (PARAMETER OR VALUE??)
        :param depth: (current depth NEEDED???)
        :return: quantum circuit corresponding to the cost hamiltonian
        """
        raise NotImplementedError

    def cost_circuit_parameterized(self, angles, depth):
        """
        Implements the function that returns the cost hamiltonian part of the PARAMETERIZED circuit

        :param angles: (PARAMETER OR VALUE??)
        :param depth: (current depth NEEDED???)
        :return: PARAMETERIZED quantum circuit corresponding to the cost hamiltonian
        """
        raise NotImplementedError
    

    def mixer_circuit(self, angles, depth):
        """
        Implements the function that returns the mixer part of the circuit

        :param angles: (PARAMETER OR VALUE??)
        :param depth: (current depth NEEDED???)
        :return: quantum circuit corresponding to the mixer
        """
        raise NotImplementedError

    def mixer_circuit_parameterized(self, angles, depth):
        """
        Implements the function that returns the mixer part of the PARAMETERIZED circuit

        :param angles: (PARAMETER OR VALUE??)
        :param depth: (current depth NEEDED???)
        :return: PARAMETERIZED quantum circuit corresponding to the mixer
        """
        raise NotImplementedError
    
    def setToInitialState(self, q):
        """
        Implements the function that sets the circuit in the initial state
        :param q: The qubit register which is initialized

        """
        raise NotImplementedError
    def computeBestMixer(self):

        raise NotImplementedError
    
    


   

################################
# generic functions
################################
    def createCircuit(self, angles, depth, draw_only = False):
        """
        implements a function to create the circuit

        :return: an instance of the qiskti class QuantumCircuit
        """
        if not self.N_beta:
            #Need self.N_beta to allocate space
            self.computeBestMixer()
        if draw_only:
            return 0
        if self.use_parameterized_circuit:
            if self.current_circuit_depth != depth:   #NEED THIS??
                self.current_circuit_depth = depth    #Needed for mixer_circuit_parameterized and/or cost_circuit_parameterized
                self.gamma_params = [None]*depth      #For ruben method we need nothing
                #self.beta_params = [None]*depth
                self.beta_params = [[None for _ in range(self.N_beta)] for _ in range(depth)] #Important to allocate space??
                q = QuantumRegister(self.N_assets)
                c = ClassicalRegister(self.N_assets)    #Do this for every depth???
                self.parameterized_circuit = QuantumCircuit(q, c)

                ### initial state
                self.setToInitialState(q) #Do this for every depth????
                for d in range(depth):
                    self.cost_circuit_parameterized(d, q)
                    self.mixer_circuit_parameterized(d, q)

                self.parameterized_circuit.measure(q, c)
                
            return self._applyParameters(angles, depth)
            
        else:              
            #Not parameterized 
            raise NotImplementedError
            cost_circuit = self.cost_circuit(angles, depth)
            mixer_circuit = self.mixer_circuit(angles, depth)
            composed_circuit = cost_circuit.compose(mixer_circuit, inplace = False)
            return composed_circuit




    def isFeasible(self, string):
        """
        needs to be implemented to run successProbability
        """
        return True

    def successProbability(self, angles, backend, shots, noisemodel=None):
        """
        success is defined through cost function to be equal to 0
        """
        depth=int(len(angles)/2)
        circ=self.createCircuit(angles, depth)
        if backend.configuration().local:
            job = execute(circ, backend, shots=shots)
        else:
            job = start_or_retrieve_job("sprob", backend, circ, options={'shots' : shots})

        jres=job.result()
        counts_list=jres.get_counts()
        if isinstance(counts_list, list):
            s_prob=[]
            for i, counts in enumerate(counts_list):
                tmp=0
                for string in counts:
                    # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                    if self.isFeasible(string[::-1]):
                        tmp+=counts[string]
                s_prob.append(tmp)
        else:
            s_prob=0
            for string in counts_list:
                # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                if self.isFeasible(string[::-1]):
                    s_prob+=counts_list[string]
        return s_prob/shots
    
    def getParametersToBind(self, angles, depth, asList=False):
        """
        Utility function to structure the parameterized parameter values 
        so that they can be applied/bound to the parameterized circuit.

        :param angles: gamma and beta values
        :param depth: circuit depth
        :asList: Boolean that specify if the values in the dict should be a list or not
        :return: A dict containing parameters as keys and parameter values as values
        """
        if self.ruben:
            #Now we use the more mixer parameters per depth
            num_mixer_params = self.N_beta*depth
            num_phase_params = depth
            assert(len(angles) == num_mixer_params + num_phase_params)
        
        else:
            #If we only use two parameters per depth
            assert(len(angles) == 2*depth) 
        
        params = {}
        #PROBLEMS HERE, angles needs changing
        for d in range(depth):
            if asList:
                params[self.gamma_params[d]] = [angles[(1+self.N_beta)*d + 0]] #make angles into a 2D array?
                
                for p in range(self.N_beta):
                    params[self.beta_params[d][p]] = [angles[(1+self.N_beta)*d +(p+1)]]
            else:
                params[self.gamma_params[d]] = angles[(1+self.N_beta)*d + 0]
                for p in range(self.N_beta):
                    params[self.beta_params[d][p]]  = angles[(1+self.N_beta)*d + (p+1)]
        return params
    
    def _applyParameters(self, angles, depth):
            """
            Wrapper for binding the given parameters to a parameterized circuit.
            Best used when evaluating a single circuit, as is the case in the optimization loop.
            """
            params = self.getParametersToBind(angles, depth)
            return self.parameterized_circuit.bind_parameters(params)   

    def loss(self, angles, backend, depth, shots, precision, noisemodel):
        """
        loss function
        :return: an instance of the qiskti class QuantumCircuit
        """
        self.g_it+=1

        circuit = None
        n_target=shots
        self.stat.reset()
        shots_taken=0
        #WHAT IS VALUE 3 IN FOR LOOP??
        for i in range(3):
            if backend.configuration().local:
                if self.use_parameterized_circuit:
                    params = self.getParametersToBind(angles, depth, asList=True)
                    job = execute(self.parameterized_circuit, 
                                  backend=backend, noise_model=noisemodel, shots=shots,
                                  parameter_binds=[params], optimization_level=0)
                else:
                    if circuit is None:
                        circuit = self.createCircuit(angles, depth)
                    job = execute(circuit, backend=backend, noise_model=noisemodel, shots=shots)
            else:
                name=""
                job = start_or_retrieve_job(name+"_"+str(opt_iterations), backend, circuit, options={'shots' : shots})
            shots_taken+=shots
            _,_ = self.measurementStatistics(job)
            if precision is None:
                break
            else:
                v=self.stat.get_Variance()
                shots=int((np.sqrt(v)/precision)**2)-shots_taken
                if shots<=0:
                    break

        self.num_shots['d'+str(self.current_depth+1)]+=shots_taken

        self.g_values[str(self.g_it)] = -self.stat.get_CVaR() 
        self.g_angles[str(self.g_it)] = angles.copy()

        #opt_values[str(opt_iterations )] = e[0]
        #opt_angles[str(opt_iterations )] = angles
        return -self.stat.get_CVaR()

    def measurementStatistics(self, job):
        """
        implements a function for expectation value and variance

        :param job: job instance derived from BaseJob
        :return: expectation and variance
        """
        jres=job.result()
        counts_list=jres.get_counts()
        if isinstance(counts_list, list):
            expectations = []
            variances = []
            for i, counts in enumerate(counts_list):
                self.stat.reset()
                for string in counts:
                    # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                    cost = self.cost(string[::-1])
                    self.stat.add_sample(cost, counts[string])
                expectations.append(self.stat.get_CVaR())
                variances.append(self.stat.get_Variance())
            return expectations, variances
        else:
            for string in counts_list:
                # qiskit binary strings use little endian encoding, but our cost function expects big endian encoding. Therefore, we reverse the order
                cost = self.cost(string[::-1])
                self.stat.add_sample(cost, counts_list[string])
            return self.stat.get_CVaR(), self.stat.get_Variance()

    def hist(self, angles, backend, shots, noisemodel=None):
        #depth=int(len(angles)/2)   #NEED THIS to change
        depth = int(len(angles)/(1+self.N_beta))
        circ=self.createCircuit(angles, depth)
        if backend.configuration().local:
            job = execute(circ, backend, shots=shots)
        else:
            job = start_or_retrieve_job("hist", backend, circ, options={'shots' : shots})
        return job.result().get_counts()

    def random_init(self, gamma_bounds,beta_bounds,depth):
        """
        Enforces the bounds of gamma and beta based on the graph type.
        :param gamma_bounds: Parameter bound tuple (min,max) for gamma
        :param beta_bounds: Parameter bound tuple (min,max) for beta
        :return: np.array on the form (gamma_1, beta_1, gamma_2, ...., gamma_d, beta_d)
        """
        gamma_list = np.random.uniform(gamma_bounds[0],gamma_bounds[1], size=depth)
        beta_list = np.random.uniform(beta_bounds[0],beta_bounds[1], size=depth)
        initial = np.empty((gamma_list.size + beta_list.size,), dtype=gamma_list.dtype)
        initial[0::2] = gamma_list
        initial[1::2] = beta_list
        return initial
    
    def initialize_angles(self, low_bound = 0, high_bound = 2*np.pi):
        if self.hot_start:
            raise NotImplementedError
        else:
            if self.randomize_init:
                gamma_val = [np.random.uniform(low_bound, high_bound)]
                print("self.N_beta: ", self.N_beta)
                beta_vals = [0]*self.N_beta
                for i in range(self.N_beta):
                    beta_vals[i] = np.random.uniform(low_bound, high_bound)
                angles0 = np.array(gamma_val + beta_vals)
            else:
                angles0 = np.zeros(1 + self.N_beta)
        return angles0




    def interp(self, angles):
        """
        INTERP heuristic/linear interpolation for initial parameters
        when going from depth p to p+1 (https://doi.org/10.1103/PhysRevX.10.021067)
        E.g. [0., 2., 3., 6., 11., 0.] becomes [2., 2.75, 4.5, 7.25, 11.]

        :param angles: angles for depth p
        :return: linear interpolation of angles for depth p+1
        """
        depth=len(angles)

        tmp=np.zeros(len(angles)+2)
        tmp[1:-1]=angles.copy()
        w=np.arange(0,depth+1)
        return w/depth*tmp[:-1] + (depth-w)/depth*tmp[1:]

    def sample_cost_landscape(self, backend, shots=1024, noisemodel=None, verbose=True, angles={"gamma": [0,2*np.pi,20], "beta": [0,2*np.pi,20]}):
        if verbose:
            print("Calculating Energy landscape for depth p=1...")
        if self.ruben:
            #Cannot calculate cost landscape with ruben method
            raise NotImplementedError
        depth=1

        tmp=angles["gamma"]
        self.gamma_grid = np.linspace(tmp[0],tmp[1],tmp[2])
        tmp=angles["beta"]                       
        self.beta_grid = np.linspace(tmp[0],tmp[1],tmp[2])

        if backend.configuration().local:

            if self.use_parameterized_circuit:
                self.createCircuit(np.array((self.gamma_grid[0],self.beta_grid[0])), depth)
                #parameters = []
                gamma = [None]*angles["beta"][2]*angles["gamma"][2]
                beta  = [None]*angles["beta"][2]*angles["gamma"][2]
            
                counter = 0
                for b in range(angles["beta"][2]):
                    for g in range(angles["gamma"][2]):
                        gamma[counter] = self.gamma_grid[g]
                        beta[counter]  = self.beta_grid[b]
                        counter += 1
                
                parameters = [{self.gamma_params[0]: gamma,
                                self.beta_params[0]: beta}]
                        
                print("Executing sample_cost_landscape")
                print("parameters: ", len(parameters), len(parameters[0][self.gamma_params[0]]))
                if (len(parameters[0][self.gamma_params[0]]) < 15):
                    print(parameters)
                job = execute(self.parameterized_circuit, backend, shots=shots, 
                                parameter_binds=parameters, optimization_level=0)
                print("Done execute")
                e, v = self.measurementStatistics(job)
                print("Done measurement")
                self.E = -np.array(e).reshape(angles["beta"][2],angles["gamma"][2])
                self.Var = np.array(v).reshape(angles["beta"][2],angles["gamma"][2])

            else: # not parameterized circuit
                circuits=[]
                for beta in self.beta_grid:
                    for gamma in self.gamma_grid:
                        circuits.append(self.createCircuit(np.array((gamma,beta)), depth))
                job = execute(circuits, backend, shots=shots)
                e, v = self.measurementStatistics(job)
                self.E = -np.array(e).reshape(angles["beta"][2],angles["gamma"][2])
                self.Var = np.array(v).reshape(angles["beta"][2],angles["gamma"][2])
        else:
            self.E = np.zeros((angles["beta"][2],angles["gamma"][2]))
            self.Var = np.zeros((angles["beta"][2],angles["gamma"][2]))
            b=-1
            for beta in self.beta_grid:
                b+=1
                g=-1
                for gamma in self.gamma_grid:
                    g+=1
                    circuit = createCircuit(np.array((gamma,beta)), depth)
                    name=""
                    job = start_or_retrieve_job(name+"_"+str(b)+"_"+str(g), backend, circuit, options={'shots' : shots})
                    e,v = self.measurementStatistics(job)
                    self.E[b,g] = -e[0]
                    self.Var[b,g] = -v[0]

        #self.current_depth=1
        if verbose:
            print("Calculating Energy landscape done")

    def get_current_deptgh(self):
        return self.current_depth

    def local_opt(self, angles0, backend, shots, precision, noisemodel=None, method='COBYLA'):
        """

        :param angles0: initial guess
        """

        depth=int(len(angles0)/(1+self.N_beta))

        self.num_shots['d'+str(self.current_depth+1)]=0
        res = minimize(self.loss, x0 = angles0, method = method,
                       args=(backend, depth, shots, precision, noisemodel))
        return res

    def increase_depth(self, backend, shots=1024, precision=None, noisemodel=None, method='COBYLA'):
        """
        sample cost landscape

        :param backend: backend
        :param shots: if precision=None, the number of samples taken
                      if precision!=None, the minimum number of samples taken
        :param precision: precision to reach for expectation value based on error=variance/sqrt(shots)
        """

        t_start = time.time()
        if self.current_depth == 0:
            if self.E is None:
                if self.ruben:
                    print("HELLO")
                    angles0 = self.initialize_angles()
                else:
                    self.sample_cost_landscape(backend, shots, noisemodel=noisemodel, angles={"gamma": [0,2*np.pi,20], "beta": [0,2*np.pi,20]})
                    ind_Emin = np.unravel_index(np.argmin(self.E, axis=None), self.E.shape)
                    angles0=np.array((self.gamma_grid[ind_Emin[1]], self.beta_grid[ind_Emin[0]]))
            self.angles_hist['d1_initial']=angles0
        else:
            angles0=np.zeros((1+self.N_beta)*(self.current_depth+1)) #new array of "initial angles"

            gamma=self.angles_hist['d'+str(self.current_depth)+'_final'][::(1 + self.N_beta)] #picks out the values for gamma 
            gamma_interp=self.interp(gamma)
            angles0[::(1+self.N_beta)]=gamma_interp

            betas = [None]*self.N_beta
            betas_interp = [None]*self.N_beta

            for p in range(self.N_beta):
                betas[p]=self.angles_hist['d'+str(self.current_depth)+'_final'][(p+1)::(1+self.N_beta)] #picks out the values for the betas
                betas_interp[p]=self.interp(betas[p])
                angles0[(p+1)::(1+self.N_beta)]=betas_interp[p]

            self.angles_hist['d'+str(self.current_depth+1)+'_initial']=angles0 #NEW ARRAY LONGER THAN FOR PREVIOUS DEPTHS?? ALSO, RIGHT STRUCTURES WITH ANGLES??

        self.g_it=0
        self.g_values={}
        self.g_angles={}

        if self.use_parameterized_circuit:
            # Make sure that we have created a parameterized circuit before calling local_opt
            self.createCircuit(angles0, int(len(angles0)/(1 + self.N_beta)))   #IS THIS CORRECT DEPTH???

        res = self.local_opt(angles0, backend, shots, precision, noisemodel=noisemodel, method=method)
        if not res.success:
            raise Warning("Local optimization was not successful.", res)
        self.num_fval['d'+str(self.current_depth+1)]=res.nfev
        self.t_per_fval['d'+str(self.current_depth+1)] = (time.time() - t_start) / res.nfev
        print("cost(depth=",self.current_depth+1,")=", res.fun)

        ind = min(self.g_values, key=self.g_values.get)
        self.angles_hist['d'+str(self.current_depth+1)+'_final']=self.g_angles[ind]
        self.costval['d'+str(self.current_depth+1)+'_final']=self.g_values[ind]


        self.current_depth+=1


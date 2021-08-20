import time
import datetime
from dimod.decorators import vartype_argument
from qiskit.finance.data_providers import RandomDataProvider, WikipediaDataProvider
import cvxpy as cp
import numpy as np
import dimod
from dwave_qbsolv import QBSolv, SOLUTION_DIVERSITY
from portfolio_4_transcost import MarkowitzWithTransactionsCost
import neal
import dimod
from dwave.system import LeapHybridSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

from mosek.fusion import *
import numpy as np

def MarkowitzWithTransaction(n,mu,GT,x0,w0,gamma,f,g):
    # Upper bound on the traded amount
    u = n*[w0]

    with Model("Markowitz portfolio with transaction costs") as M:
        #M.setLogHandler(sys.stdout)

        # Defines the variables. No shortselling is allowed.
        x = M.variable("x", n, Domain.greaterThan(0.0))

        # Additional "helper" variables 
        z = M.variable("z", n, Domain.unbounded())   
        # Binary variables
        y = M.variable("y", n, Domain.binary())
        t = M.variable("t", 1, Domain.unbounded())

        #  Maximize expected return
        M.objective('obj', ObjectiveSense.Minimize, Expr.add(t, Expr.dot(-1 * mu,x)))

        # Invest amount + transactions costs = initial wealth
        M.constraint('budget', Expr.add([Expr.sum(x), Expr.dot(f,y),Expr.dot(g,z)] ), Domain.equalsTo(w0))

        # TODO: change!1
        M.constraint('risk', Expr.vstack(t, 2*gamma, Expr.mul(GT,x)), Domain.inRotatedQCone())
        # Constraints for turning y off and on. z-diag(u)*y<=0 i.e. z_j <= u_j*y_j
        M.constraint('y_on_off', Expr.sub(z,Expr.mulElm(u,y)), Domain.lessThan(0.0))

        # Integer optimization problems can be very hard to solve so limiting the 
        # maximum amount of time is a valuable safe guard
        M.setSolverParam('mioMaxTime', 30.0) 
        M.solve()

        return x.level(), y.level(), z.level() 

if __name__ == '__main__':    

    xsol, ysol, zsol = MarkowitzWithTransactionsCost(n,mu,GT,x0,w,gamma,f,g)
    print("\n---------------------Cj--------------------------------------------------------------");
    print('Markowitz portfolio optimization with transactions cost')
    print("-----------------------------------------------------------------------------------\n");
    print('Expected return: %.4e Std. deviation: %.4e Transactions cost: %.4e' % \
          (np.dot(mu,xsol),gamma, np.dot(f,ysol)+np.dot(g,zsol)))

    def solve_sparsity(self, w, method = 'annealing', sparse_constr = True, solver = 'MOSEK', lda = -1, lookahead = False, exact = None, sampler = None, testing = [0,0]):
        w_old = w
        if sparse_constr:
            sigma_temp = np.array([self.sigma[i] for i in range(len(w)) if w[i]])
            sigma = np.array([sigma_temp[:, i] for i in range(len(w)) if w[i]])
            mu = np.array([self.mu[i] for i in range(len(w)) if w[i]])
            w = np.array([w[i] for i in range(len(w)) if w[i]])
        else:
            sigma = self.sigma
            mu = self.mu

        sigma_p = (sigma * w).T * w
        sigma_p_old = np.array(sigma_p)
        mu_p = -self.gamma * mu * w
        mu_p_old = np.array(mu_p)
        l = len(w)

        if method == 'circuit':
            return NotImplementedError
        elif method == 'annealing':


            #print(sigma)
            #print(w)
            #print(2 * sigma @ w)
            w_total = self.lc * w + self.fc * np.ones((l,))

            # we subtract the entire thing eventually so

            # linear terms
            # lin_penalty = 2 * self.B * lda
            qubo_lin = {i:mu_p[i] for i in range(l)}
            # quadratic terms
            qubo_quad = {}
            for i in range(l):
                qubo_quad.update({(i,j):(sigma_p[i][j]) for j in range(l) if sigma_p[i][j] != 0})

            self.bqm = dimod.BQM(qubo_lin, qubo_quad, vartype = dimod.BINARY)

            if exact is None:
                exact = l < 16
            if exact:
                print("Problem is small. Using exact solver")
                sampler = dimod.ExactSolver()
                sampleset = sampler.sample(self.bqm).aggregate()
            else:
                print("Problem is large. Using simulated annealer")
                #self.sampler = dimod.ExactSolver()
                if sampler is None:
                    #self.sampler = EmbeddingComposite(DWaveSampler())
                    self.sampler = neal.SimulatedAnnealingSampler()
                else:
                    self.sampler = sampler
                #self.sampler = LeapHybridSampler()    
                sampleset = self.sampler.sample(self.bqm, num_reads = 512).aggregate()
                #initial_states=(np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
                #dtype = 'int8'))).aggregate()
                # for i in range(10):
                # sampleset = QBSolv().sample(self.bqm,
                # solver=self.sampler, num_repeats = 200, algorithm =
                # SOLUTION_DIVERSITY, verbosity = 2).aggregate()
                # print(sampleset.data_vectors['energy'])

            best = [sampleset.first.sample[i] for i in range(l)]
            print(best)
            error = []
            base_error = []

                print(f"Sparsity test average error is {100 *np.mean(np.abs(error))}% with lookahead and {100 * np.mean(np.abs(base_error))}% without")

            self.err = error
            self.quad = qubo_quad
            self.lin = qubo_lin
            self.sig = sigma_p
            self.sig2 = sigma_p_old
            self.m = mu_p
            self.m2 = mu_p_old
            return sampleset

                
        elif method == 'classical':
            v = cp.Variable((len(w),), boolean=True)
            constr = [self.lc * (w.T @ v) + self.fc * cp.norm1(v) <= self.B]
            obj = cp.quad_form(v, sigma_p) - self.gamma * mu_p.T @ v

            problem = cp.Problem(cp.Minimize(obj), constr)
            problem.solve(solver = solver)

            if sparse_constr:
                # construct original vector
                v_new = np.zeros((len(w_old),))
                count = 0
                for i in range(len(w_old)):
                    if w_old[i]:
                        v_new[i] = np.round(v.value[count])
                        count += 1
            else:
                v_new = v

            return v_new
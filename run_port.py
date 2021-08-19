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

# TODO: Comments, clean code, incorporate logger (esp verbose)

class card_portfolio_problem():
    def __init__(self, num_companies = 20, fixed_cost = 0.5, linear_cost = 1,
        budget = 20, gamma = 0.5, data = None, mu = None, sigma = None, k = 6,
        use_tickers = False):

        """Need docstring"""
        self._num_companies = num_companies
        self._fixed_cost = fixed_cost
        self._linear_cost = linear_cost
        self._budget = budget
        self._gamma = gamma
        self._data = data
        self._mu = mu
        self._sigma = sigma
        self._k = k
        self._use_tickers = use_tickers

        # 46 companies
        default_tickers = ['AAPL','MSFT','GOOG','GOOGL','AMZN','FB' \
            ,'V','NVDA','JPM','JNJ','WMT','UNH','MA','HD','PG','BAC','PYPL', \
            'DIS','ASML','ADBE','CMCSA','NKE','TM','KO','XOM','ORCL','PFE','CRM', \
            'CSCO','LLY','VZ','NFLX','INTC','PEP','ABT','DHR','NVO','TMO','NVS', \
            'ABBV','ACN','T','AVGO','BHP','CVX','MRK']
        if use_tickers:
            self._companies = default_tickers[:num_companies]
        else:
            self._companies = [i for i in range(self._num_companies)]
        self.errors = []
            
    @property
    def n(self) -> int:
        """Return the number of companies in the problem."""
        return self._num_companies

    @property
    def fc(self) -> int:
        """Return the fixed cost."""
        return self._fixed_cost

    @property
    def lc(self) -> int:
        """Return the linear cost."""
        return self._linear_cost

    @property
    def B(self) -> int:
        """Return the budget."""
        return self._budget

    @property
    def gamma(self) -> int:
        """Return gamma, the risk aversion factor."""
        return self._gamma

    @property
    def companies(self) -> list:
        """Return a list of company names."""
        return self._companies

    @property
    def data(self) -> np.ndarray:
        """Return gamma, the risk aversion factor."""
        return self._data

    @property
    def mu(self) -> np.ndarray:
        """Return the budget."""
        return self._mu

    @property
    def sigma(self) -> np.ndarray:
        """Return the risk matrix."""
        return self._sigma

    @property
    def k(self) -> int:
        """Return the size of the beam search."""
        return self._k

    # use ibm's random data generator to get data. compute mean and covariance matrices
    def get_data_variables(self, seed = None):
        key = '3KPsx3c_CgvUgDEcxuh4'
        if self._use_tickers:
            data_provider = WikipediaDataProvider(tickers=self.companies,
                start=datetime.datetime(2018, 1, 1), end=datetime.datetime(2018,
                3, 30))
        else:
            if seed is None:
                data_provider = RandomDataProvider(tickers=self.companies,
                start=datetime.datetime(2018, 1, 1), end=datetime.datetime(2018,
                3, 30))
            else:
                data_provider = RandomDataProvider(tickers=self.companies,
                start=datetime.datetime(2018, 1, 1), end=datetime.datetime(2018,
                3, 30), seed=seed)
        self.dataprovide = data_provider
        data_provider.run()
        self._data = np.array(data_provider._data)
        self._mu = np.array(data_provider.get_period_return_mean_vector())
        self._sigma = np.array(data_provider.get_period_return_covariance_matrix())

    # solve convex relaxation to get initial point
    def solve_relaxed_problem(self, margin):
        x = cp.Variable((self.n,))
        B = self.B - self.n*margin
        assert B > 0, "Error: modified budget must be positive"
        # removed margin of from adding to lin cost (self.fc/max_val)
        constr = [(self.lc) * cp.norm(x,1) <= B]
        constr += [x >= 0]
        obj = cp.quad_form(x, self.sigma) - self.gamma * (self.mu.T @ x)
        problem = cp.Problem(cp.Minimize(obj), constr)
        problem.solve()
        #return self.convex_projection(x.value)
        return x.value


    # solve for weights w given sparsity v
    def solve_weights(self, v, sparse_constr = True):
        if sparse_constr:
            sigma_temp = np.array([self.sigma[i] for i in range(len(v)) if v[i]])
            sigma = np.array([sigma_temp[:, i] for i in range(len(v)) if v[i]])
            mu = np.array([self.mu[i] for i in range(len(v)) if v[i]])
        else:
            sigma = self.sigma
            mu = self.mu
        w = cp.Variable((sigma.shape[0],))

        # Emphasize that this is a constant
        fix_cost = self.fc*sum(v)
        constr = [self.lc*cp.norm1(w) + fix_cost  <= self.B]
        constr += [w >= 0]
        obj = cp.quad_form(w, sigma) - self.gamma * mu.T @ w
        problem = cp.Problem(cp.Minimize(obj), constr)
        problem.solve()

        if sparse_constr:
            # reconstruct soln
            count = 0
            w_new = np.zeros((len(v),))
            for i in range(len(v)):
                if v[i]:
                    w_new[i] = w[count].value
                    count += 1
        else:
            w_new = w.value

        return w_new

    #solve for sparsity v given weights w. sparse_constr True means that we only
    #optimize over the v_i's where w_i is nonzero
    # lda is hyperparam for weighing constr (only needed for annealing)
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

            # inequality constraint
            # num_constr = int(np.ceil(np.log(np.sum(np.sort(w_p)[-2:]))/np.log(2)))
            # c = np.r_[w_p, np.array([1 << i for i in range(num_constr)])]
            # sigma_p = np.r_[np.c_[sigma_p,np.zeros((l,num_constr))], np.zeros((num_constr, l + num_constr))]
            # sigma_p += lda * np.outer(c,c)
            # l += num_constr
            
        
            # precompute
            # print(c)
            # mod_c = c * lda
            # mod_w = lda * w_p


            #print(sigma)
            #print(w)
            #print(2 * sigma @ w)
            w_total = self.lc * w + self.fc * np.ones((l,))
            if lookahead:
                abs_grad = np.abs(2 * sigma @ w - self.gamma * mu)
                #print(self.gamma * mu)
                print(abs_grad)
                #print(w_total)

                # this is just the cardinality of the original w
                if lda == -1:
                    lda = 1 / (l - 1)

                sigma_p += lda * np.outer(abs_grad, w_total)
                mu_p -= lda * self.B * abs_grad

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
            if testing[0]:
                for i in range(testing[0]):
                    perm = np.random.permutation(l)[:testing[1]]
                    perturb = np.array(best)
                    for j in perm:
                        perturb[j] ^= 1
                    print(perturb)
                    perturb_a = to_array(perturb, w_old)
                    true_score = self.eval_score(self.solve_weights(perturb_a), perturb_a)
                    print(true_score)
                    score = perturb.T @ sigma_p @ perturb + perturb.T @ mu_p
                    print(score)
                    base_score = perturb.T @ sigma_p_old @ perturb + perturb.T @ mu_p_old
                    print(base_score)
                    error += [(true_score - score)/(true_score)]
                    base_error += [(true_score - base_score)/(true_score)]

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

    def convex_projection(self, x):
        top_args = np.flip(np.argsort(x))
        result = np.zeros((len(x),))
        B_so_far = 0
        for i in range(len(x)):
            B_so_far += self.fc * card(result[top_args[i]]) + self.lc * result[top_args[i]]
            if B_so_far <= self.B:
                result[top_args[i]] = x[top_args[i]]
            elif self.B - B_so_far > self.fc:
                result[top_args[i]] = (x[top_args[i]] - self.fc) / self.lc
            else:
                break
        return result

    # Prioritize readability over speed when checking constriants since only
    # users check will call with checks
    def eval_score(self, w, v, checks = False, verbose = False, lookahead = False):
        if checks:
            # Check constraints
            assert np.all([elem >= 0 for elem in w]), "Weight vector w does not satisfy w >= 0."
            assert np.all([elem * (1 - elem) == 0 for elem in v]), "Sparsity vector v is not binary."
            assert np.all([w[i] == 0 or v[i] == 1 for i in range(len(w))]), "Entries \
            of w can only be nonzero when the corresponding entry of v is 1."
            assert self.lc * np.sum(w) + self.fc * card(v) <= self.B, f"Solution goes over budget\
             ({self.lc * np.sum(w) + self.fc * card(v)} > {self.B})."

        # Return score
        # w =  [w[i]if v[i] else 0 for i in range(len(w))]
        # if lookahead:
        #     abs_grad = np.abs(2 * self.sigma @ w - self.gamma * self.mu)

        #     # this is just the cardinality of the original w
        #     lda = 1 / len(v)

        #     sigma += lda * np.outer(abs_grad, w_total)
        #     mu -= lda * self.B * abs_grad
        cost = w @ self.sigma @ w - self.gamma * (self.mu.T @ w)
        budget = self.lc * np.sum(w) + self.fc * card(v)
        if verbose:
            print(f"Solution cost: {cost}\
            \nSpends {budget} of {self.B} budget.")
        
        return cost
        
    def run_iter(self, w, verbose = False, lookahead = True, exact = False):
        size_mult = 5
        # try just concatenating samplesets here
        sampleset = self.solve_sparsity(w, method = "annealing", sparse_constr = True, lookahead = lookahead, exact=  exact)
        best_set = []
        count = 0
        print(sampleset)
        for sample, energy in sampleset.data(fields = ['sample', 'energy'], sorted_by='energy'):
            #print((sample, energy))

            
            sample = to_array(sample, w)
            prev_energy = self.eval_score(w, sample)
            best_set += [(self.solve_weights(sample, sparse_constr = True), sample, energy, prev_energy)]
            #self.eval_score(best_set[-1][0], sample, verbose = True)
            count += 1
            if count == size_mult * self.k:
                break
        
        print(f'eval len is {len(best_set)}')
        evals = [self.eval_score(best_set[i][0], best_set[i][1]) for i in range(len(best_set))]
        best_indices = np.argsort(evals)[:min(self.k, len(best_set))]
        #best_set = best_set[list(best_indices)]
        new_set = []
        new_evals = []
        for i in best_indices:
            new_set += [best_set[i]]
            new_evals += [evals[i]]

        best_set = new_set
        evals = new_evals

        if verbose:
            if (len(best_indices) == self.k):
                percent_top = 100 * (self.k - sum([1 for i in range(self.k) if best_indices[i] >= self.k])) / self.k
                print(f"{percent_top}% of samples predicted in top k ended up in top k.")

            errs = [100 * (evals[i] - best_set[i][2]) / evals[i] for i in range(len(best_set))]
            errs_no_lookahead = [100 * (evals[i] - best_set[i][3]) / evals[i] for i in range(len(best_set))] 
            avg_error = np.mean(errs)
            self.errors += errs
            print(f"Average approximation error is {avg_error}% with lookahead and {np.mean(errs_no_lookahead)}% without.")
        
        for i in range(len(best_set)):
            best_set[i] = (best_set[i][0], best_set[i][1], evals[i])

        return best_set, evals




def card(x):
    if isinstance(x, np.ndarray):
        return sum([1 for elem in x if elem != 0])
    else:
        return x != 0

#x should be an aggregated sampleset
def top_k(x, k):
    assert len(x) > k, "The number of selected components is greater than the number of unique samples."
    # check if most sampled are lowest energy
    elems = [x[i][2] for i in range(len(x))]
    return x[np.argsort(elems)][-k:]

# takes in dict of elems to vals from sampleset
def to_array(x, w):
    count = 0
    result = np.zeros((len(w),))
    for i in range(len(w)):
        if w[i]:
            result[i] = x[count]
            count += 1
    assert count != len(x) - 1, "Error: x doesn't fill out w's sparsity pattern"
    return result


def run_quantum(verbose = False, **kwargs):
    margin = -.5
    max_val = 8
    problem = card_portfolio_problem(**kwargs)
    problem.get_data_variables(seed = 1234)
    if verbose:
        print("Getting initial point")

    x_init = problem.solve_relaxed_problem(margin, max_val)
    w = np.array([x_init[i] if x_init[i] > 0.01 else 0 for i in range(len(x_init))])
    temp = problem.convex_projection(w)
    v = np.array([1 if temp[i] !=0 else 0 for i in range(len(temp))])
    if verbose:
        problem.eval_score(w,v, checks=False)

    iter = 0
    prev_best = 100

    print("Computing initial set.")
    best_set, _= problem.run_iter(w, verbose = verbose)
    while True:
        iter += 1
        if verbose:
            print(f"Iteration {iter}:")

        new_solns = {}
        all_energies = set()
        for (w, v, _) in best_set:
            candidates, energies = problem.run_iter(w, verbose = False)
            #if verbose:
            #    print(f"Energies are {energies}")

            for i in range(len(energies)):
                new_solns[energies[i]] = candidates[i]
                all_energies.add(energies[i])
        all_energies = list(all_energies)
        best_energies = np.sort(all_energies)[:problem.k]

        best_solns = []
        for energy in best_energies:
            best_solns += [new_solns[energy]]
            
        print(f"best soln is {best_solns[0]}")
        print(f"Best solution has cost {best_solns[0][2]} and invests in \
            {sum(best_solns[0][1])} assets ({100 * sum(best_solns[0][1]/problem.n)}%).")
        
        print(best_energies)
        if best_solns[0][2] >= prev_best:
            break
        else:
            prev_best = best_solns[0][2]
            print('here we go again')
        
        best_set = best_solns
    
    print("Algorithm converged.")

def run_classical(verbose = False, **kwargs):
    margin = 0.2
    problem = card_portfolio_problem(**kwargs)
    problem.get_data_variables(seed = 1234)
    if verbose:
        print("Getting initial point")

    x_init = problem.solve_relaxed_problem(margin)
    w = np.array([x_init[i] if x_init[i] > 0.01 else 0 for i in range(len(x_init))])
    temp = problem.convex_projection(w)
    v = np.array([1 if temp[i] !=0 else 0 for i in range(len(temp))])
    if verbose:
        problem.eval_score(w,v, checks=False)

    iter = 0
    prev_best = 100
    print(len(v))
    while True:
        iter += 1
        if verbose:
            print(f"Iteration {iter}:")

        v_new = problem.solve_sparsity(w, method = "classical", sparse_constr = True)
        print(len(v_new))
        w_new = problem.solve_weights(v_new, sparse_constr = True)
        print(len(w_new))
        score =  problem.eval_score(w_new, v_new, verbose = True)

        if score < prev_best:
            print("here we go again")
            prev_best = score
        else:
            break

    print("Algorithm converged.")
    return (v_new, w_new, score)

def full_classical(**kwargs):
    problem = card_portfolio_problem(**kwargs)
    problem.get_data_variables(seed = 1234)
    a = time.time()
    n = problem.n
    x = cp.Variable(n)
    y = cp.Variable(n, boolean = True)
    constr = [x >= 0]
    constr += [x <= problem.B * y]
    constr += [problem.lc * cp.norm(x, 1) + problem.fc * cp.norm(y, 1) <= problem.B]
    obj = cp.quad_form(x, problem.sigma) - problem.gamma * (problem.mu.T @ x)
    cp_problem = cp.Problem(cp.Minimize(obj), constr)
    cp_problem.solve(solver="MOSEK")
    print(f"Total time: {time.time() - a}")
    return x.value, y.value
    
    # LT = np.linalg.cholesky(problem.sigma).T
    # return MarkowitzWithTransactionsCost(problem.n, problem.mu, LT,
    # np.zeros((problem.n,)), problem.B, problem.gamma, problem.n * [problem.fc],
    # problem.n * [problem.lc])

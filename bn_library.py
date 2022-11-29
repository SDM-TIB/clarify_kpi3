import bnlearn as bn
from matplotlib.pyplot import axis
# from importlib_metadata import Prepared
# from matplotlib.pyplot import axis
from pgmpy.estimators import K2Score, BicScore, BDeuScore

from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
import random
import pyAgrum as gum

################################################
# Structure Learning
################################################

score_funs = {'bic': BicScore, 'k2': K2Score, 'bdeu': BDeuScore}

# 1. score based
def exhaustive_search(data, score_option='bic'):
    from pgmpy.estimators import ExhaustiveSearch
    assert score_option in score_funs.keys()
    es = ExhaustiveSearch(data, scoring_method=score_funs[score_option](data))
    best_model = es.estimate()
    return best_model.edges()
    # print(best_model.edges())

def hill_climate(data, score_option='bic'):
    # based on simulated annealing
    # from pgmpy.estimators import HillClimbSearch
    assert score_option in score_funs.keys()
    model = bn.structure_learning.fit(data, methodtype='hc', scoretype='bic')
    model = bn.independence_test(model, data, alpha=0.05, prune=True)
    # best_model = HillClimbSearch(data).estimate(scoring_method=score_funs[score_option](data))
    return model['model_edges'] #.edges()

def TAN(data, root_node, score_option='bic'):
    '''Tree-augmented Naive Bayes, a tree based structure
    '''
    assert score_option in score_funs.keys()
    model = bn.structure_learning.fit(data, methodtype='tan', class_node=root_node, scoretype=score_option)
    model = bn.independence_test(model, data, alpha=0.05, prune=True)   # use chi-squre test to remove no significant edges
    return model['model_edges']

def K2(data, score_option='bic'):
    # reference https://agrum.gitlab.io/articles/agrumpyagrum-0229-and-dataframe.html
    s_learner = gum.BNLearner(data)  # creates a learner by passing the dataframe
    # s_learner.useGreedyHillClimbing()     # sets a local-search algorithm for the structural learning
    s_learner.useK2(random.shuffle([i for i in range(len(data.columns))]))  # using random the typology order
    if score_option == 'bic':
        s_learner.useScoreBIC()              # sets BIC score as the metric
    elif score_option == 'k2':
        s_learner.useScoreK2()
    elif score_option == 'bdeu':
        s_learner.useScoreBDeu()
    else:
        raise Exception("wrong score option")
    structure_learn = s_learner.learnBN()       # learning the structure
    id2name = {structure_learn.idFromName(node_n): node_n for node_n in structure_learn.names()}
    return [(id2name[ele[0]], id2name[ele[1]]) for ele in structure_learn.arcs()]

    # print(structure.arcs())
    # print(structure.nodes())
    # # print(train_data.columns)
    # print(structure.names())
    # # print(structure.cpt('Rain'))
    # print(structure.idFromName('Rain'))
    # structure.dag()

def mcmc_edges(a):
    '''a is a str representing the adjacent matrix'''
    brief = {'M':'Mutation', 'FN':'FamilyTypeNum', 'R':'Relapse', 'A': 'Age', 'C':'Cancer', 
            'F':'Family', 'S1' :'Stage', 'G': 'Gender', 'S':'Smoker'}
    long_s = {v:k for k, v in brief.items()}
    b = a.split()
    N = 9
    import numpy as np
    import pandas as pd
    b = np.array([int(b[i*N+j]) for i in range(N) for j in range(N)]).reshape((N,N))

    df = pd.read_csv('query4R.csv')
    print(df.columns)

    cols = df.columns.to_list()
    print(cols)

    edges = []
    for i in range(N):
        for j in range(N):
            if b[i][j] == 1:
                edges.append((cols[i], cols[j]))
    edges = [(long_s[ele[0]],long_s[ele[1]]) for ele in edges]
    return edges

# 2. constraint based
def constraint_based(data):
    # Structure learning
    model = bn.structure_learning.fit(data, methodtype='cs')
    model = bn.independence_test(model, data, alpha=0.05, prune=True)
    return model['model_edges']

# 3. hybrid method
def mmhc(data, score_option='bic'):
    from pgmpy.estimators import MmhcEstimator
    from pgmpy.estimators import HillClimbSearch
    assert score_option in score_funs.keys()
    mmhc = MmhcEstimator(data)
    skeleton = mmhc.mmpc()
    print("Part 1) Skeleton: ", skeleton.edges())

    # use hill climb search to orient the edges:
    model = HillClimbSearch(data).estimate(scoring_method=score_funs[score_option](data), tabu_length=10, white_list=skeleton.to_directed().edges())
    # model = bn.independence_test(model, data, alpha=0.05, prune=True)
    # return model['model_edges']
    return list(model.edges())

################################################
# Parameter Learning
################################################
# 1. Maximum Likelihood Estimator
def MLE(skeleton, data):
    from pgmpy.estimators import MaximumLikelihoodEstimator
    model = BayesianModel(skeleton)
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model

# 2. Bayesian Estimator
def BE(skeleton, data):
    from pgmpy.estimators import BayesianEstimator
    model = BayesianModel(skeleton)
    model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")
    
    return model


################################################
# Predition and Inference
################################################

def inference(model, query_nodes, evidences):
    '''model: Bayesian Network
    '''
    infer = VariableElimination(model)
    infer.query(query_nodes, evidence=evidences)

def predict(model, Y, evidences):
    '''model: Bayesian Network
    '''
    infer = VariableElimination(model)
    infer.map_query([Y], evidences)

################################################
# Intervention
################################################

def do(model, interventions):
    from pgmpy.factors.discrete.CPD import TabularCPD
    """ 
    Implement an ideal intervention for discrete variables. Modifies pgmpy's
    `do` method so it is a `do`-operator, meaning a function that takes in a
    model, modifies it with an ideal intervention, and returns a new model.
    Note that this code would need to be modified to work for continuous
    variables.
    """
    def _mod_kernel(kernel, int_val):
        """
        Modify a causal Markov kernel so all probability is on the state fixed
        by the intervention.
        """ 
        var_name = kernel.variable
        card = kernel.get_cardinality([var_name])[var_name]
        states = [kernel.get_state_names(var_name, i) for i in range(card)]
        non_int_states = set(states) - {int_val,}
        unordered_prob_vals = [[1.0]] + [[0.0] for _ in range(card - 1)]
        unordered_states = [int_val] + list(non_int_states)
        # Reorder so it matches original
        dict_ = dict(zip(unordered_states, unordered_prob_vals))
        ordered_prob_values = [dict_[k] for k in states]
        intervention_kernel = TabularCPD(
            var_name, card, ordered_prob_values,
            state_names = {var_name: states}
        )
        return intervention_kernel

    kernels = {kern.variable: kern for kern in model.get_cpds()}
    new_model = model.copy()
    for var, int_val in interventions.items():
        new_model = new_model.do(var)
        new_kernel = _mod_kernel(kernels[var], int_val)
        new_model.add_cpds(new_kernel)
    return new_model

def causal_inference(model, query_nodes, interventions, evidences={}):
    '''model: Bayesian Network
    '''
    # reference on https://colab.research.google.com/drive/1k8eoFkHugorOrjiH57bMXF84bV3ojzOm?usp=sharing#scrollTo=ZNBa1oGIOpyP 
    # from pgmpy.inference import CausalInference 
    # return CausalInference(model)
    modified_model = do(model, interventions)
    all_evidences = {**interventions, **evidences}
    infer = VariableElimination(modified_model)
    inference(infer, query_nodes, all_evidences)


################################################
# Criteria
################################################

def get_y_and_pred(model, Y, data):
    y = data[Y].values.tolist()
    data=data.drop(Y, axis=1)
    pred = model.predict(data)[Y].values.tolist()
    # infer = VariableElimination(model)

    # y = data[Y].values.tolist()
    # y_dict = {e: i for i, e in enumerate(set(y))}   # for digitalize value
    # assert len(y_dict.keys()) == 2

    # data = data.drop(Y, axis=1)
    # pred = [infer.map_query([Y], row.to_dict())for _, row in data.iterrows()]

    # y = [y_dict[e] for e in y]
    # pred = [y_dict[e[Y]] for e in pred]
    return y, pred

def log_likelihood(model, data):
    from pgmpy.metrics import log_likelihood_score
    return log_likelihood_score(model, data)

def auc(model, Y, data):
    from sklearn import metrics
    y, pred = get_y_and_pred(model, Y, data)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    return metrics.auc(fpr, tpr)

def accuracy(model, Y, data):
    from sklearn.metrics import accuracy_score
    y, pred = get_y_and_pred(model, Y, data)
    return accuracy_score(y, pred)


from pyvlmc.contexttree import ContextTreeNodeT
import numpy.random as rnd


"""
dir_par: parameter for the Dirichlet distribution.
         used to randomly generate next state probaility distributions
"""

def ProbNode(prob_dist, children={}):
    totalcount = 0
    alphabet = frozenset(prob_dist.keys())
    counts = {c: 0 for c in alphabet}
    return ContextTreeNodeT(counts, totalcount, children, alphabet, prob_dist)


def next_state_dist(alphabet, dir_par=None):
    if dir_par is None:
        dir_par = [1]*len(alphabet)
    probs = rnd.dirichlet(dir_par)
    return {c: probs[i] for i, c in enumerate(alphabet)}


def gen_binom_random_vlmc(mindepth, maxdepth, alphabet, p, dir_par=None):
    if dir_par is None:
        dir_par = [1]*len(alphabet)
    next_state_pdist = next_state_dist(alphabet, dir_par)
    if maxdepth <= 0:
        return ProbNode(next_state_pdist)
    children = {}
    for c in alphabet:
        if mindepth > 0 or rnd.random() < p:
            children[c] = gen_binom_random_vlmc(mindepth - 1, maxdepth - 1, alphabet, p, dir_par)
    for a in alphabet:
        if a not in children.keys():
            children[a] = ProbNode(next_state_pdist)
    return ProbNode(next_state_pdist, children)


def long_branch_rand_d(sym, alphabet, depth, dir_par=None):
    next_state_p = next_state_dist(alphabet, dir_par)
    if depth <= 0:
        return ProbNode(next_state_p)
    pn = ProbNode(next_state_p, {sym: long_branch_rand_d(sym, alphabet, depth-1, dir_par)})
    for a in alphabet:
        if a != sym:
            pn.children[a] = ProbNode(next_state_p)
    return pn

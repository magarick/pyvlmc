from __future__ import division
#from pyvlmc.vlmc import all_contexts, predict_next, get_context, context_marginal_probs
import pyvlmc as vl
from collections import namedtuple
import pandas as pd
import numpy as np

EmbeddedVLMC = namedtuple("EmbeddedVLMC", ["mx", "state_to_ix", "ix_to_state"])

def forward(f_hat, a_hat, p_hat, p_hat_emb, real_states, pi_hat=None):
    if pi_hat == None:
        pi_hat = p_hat_emb
    alpha = np.zeros((len(f_hat), a_hat.shape[0]))
    f_over_p = np.array(f_hat.ix[:, real_states]) / np.array(p_hat[real_states])
    alpha[0, :] = pi_hat * np.array(f_over_p[0,:])
    alpha[0, :] /= np.sum(alpha[0, :])
    for t in xrange(1, f_over_p.shape[0]):
        alpha[t, :] = np.multiply(alpha[t-1, :] * a_hat, f_over_p[t, :])
        alpha[t, :] /= np.sum(alpha[t, :])
    return alpha

def backward(f_hat, a_hat, p_hat, p_hat_emb, real_states, pi_hat=None):
    if pi_hat == None:
        pi_hat = p_hat_emb
    beta = np.zeros((len(f_hat), a_hat.shape[0]))
    T = len(f_hat)
    beta[T-1, :] = 1
    f_p = np.array(f_hat.ix[:, real_states]) / np.array(p_hat[real_states])
    for t in xrange(T-2, -1, -1):
        beta[t, :] = np.dot(a_hat, np.multiply(beta[t+1, :], f_p[t + 1, :]))
        beta[t, :] /= np.sum(beta[t, :])
    return beta

def fb_state_probabilities(alpha, beta):
    state_probs = alpha * beta
    state_probs /= np.sum(state_probs, 1)[:, np.newaxis]
    return state_probs

def fb_probs_to_original_states(fb_probs, real_states):
    original_state_prob_dict = {x:0 for x in set(real_states)}
    for i, x in enumerate(real_states):
        original_state_prob_dict[x] += fb_probs[:, i]
    return pd.DataFrame(original_state_prob_dict)

def embed_vlmc_dict(mc, interior=True):
    """Return the transitions between terminal nodes of a VLMC as a dict of dicts
    where the final values are transition probabilities"""
    contexts = [x for x in vl.all_contexts(mc, interior=interior) if len(x) > 0]
    predictions = {c: vl.predict_next(mc, c) for c in contexts}
    return {c:{vl.get_context(mc, c+(c1,), True):p for c1, p in predictions[c].items() if p > 0}
        for c in contexts}

#def embed_vlmc_dict1(mc, interior=True, path=[]):
#    """Return the transitions between terminal nodes of a VLMC as a dict of dicts
#    where the final values are transition probabilities"""
#    if not mc.children:
#        return {tuple(reversed(path)):mc.prob_dist}
#    interior_transitions = {}
#    if interior and path:
#        interior_transitions.update({tuple(reversed(path)):mc.prob_dist})
#    for k, v in mc.children.iteritems():
#        interior_transitions.update(embed_vlmc_dict1(v, interior, path + [k]))
#    if not path:
#        for k, v in interior_transitions.iteritems():
#            interior_transitions[k] = {get_context(mc, k + (v1,)):v2 for v1,v2 in v.items()}
#    return interior_transitions

def embed_vlmc(mc, interior=True):
    emb_dict = embed_vlmc_dict(mc, interior)
    N = len(emb_dict)
    emb_matrix = EmbeddedVLMC(np.matrix(np.zeros((N, N), dtype=np.float64)), {}, {})
    emb_matrix.ix_to_state.update(dict(enumerate(emb_dict.iterkeys())))
    emb_matrix.state_to_ix.update({v:k for k, v in emb_matrix.ix_to_state.iteritems()})
    for k, t in emb_dict.iteritems():
        for l, v in t.iteritems():
            emb_matrix.mx[emb_matrix.state_to_ix[k], emb_matrix.state_to_ix[l]] = v
    return emb_matrix

#def state_to_ix_and_back(states):
#    state_to_ix, ix_to_state = {}, {}
#    terminals = []
#    for i, s in enumerate(states):
#        ix_to_state[i] = s
#        state_to_ix[s] = i
#        terminals.append(s[-1])
#    return ix_to_state, state_to_ix, terminals
#
#def embedded_model_to_mx(emb_model, state_to_ix):
#    N = len(emb_model)
#    mx = np.zeros((N, N), dtype=np.float64)
#    for k in emb_model.keys():
#        for l, v in emb_model[k].iteritems():
#            mx[state_to_ix[k], state_to_ix[l]] = v
#    return np.matrix(mx)
#def vlmc_to_fb_parameters(mc):
#    embedded = embed_vlmc(mc, interior=True)
#    ix_to_state, state_to_ix, ix_to_term = state_to_ix_and_back(embedded.keys())
#    a_hat = embedded_model_to_mx(embedded, state_to_ix)
#    p_hat_emb = marginal_context_probs(mc, state_to_ix)
#    p_hat = pd.Series({k:v/mc.totalcount for k,v in mc.counts.iteritems()})
#    return (a_hat, p_hat, p_hat_emb, ix_to_term)

def marginal_context_probs(model, state_to_ix, interior=True):
    prob_d = vl.context_marginal_probs(model, interior=interior)
    n_states = len(prob_d)
    prob_arr = np.zeros(n_states - 1) if interior else np.zeros(n_states)
    for k, v in prob_d.iteritems():
        if len(k) > 0:
            prob_arr[state_to_ix[k]] = v
    return prob_arr

def vlmc_to_fb_parameters(mc, interior=True):
    embedded = embed_vlmc(mc, interior)
    p_hat_emb = marginal_context_probs(mc, embedded.state_to_ix, interior)
    p_hat = pd.Series(mc.prob_dist)
    ix_to_term = [x[1][-1] for x in sorted(embedded.ix_to_state.items())]
    return (embedded.mx, p_hat, p_hat_emb, ix_to_term)

def run_fb(f_hat, a_hat, p_hat, p_hat_emb, ix_to_term):
    alpha = forward(f_hat, a_hat, p_hat, p_hat_emb, ix_to_term)
    beta = backward(f_hat, a_hat, p_hat, p_hat_emb, ix_to_term)
    gamma = fb_state_probabilities(alpha, beta)
    return fb_probs_to_original_states(gamma, ix_to_term)


from __future__ import division
from math import log
from operator import itemgetter
from scipy.stats import chi2
import numpy as np
import random
from contexttree import ContextTreeNode
import bisect
import heapq as hq

###Functions to related to fitting a VLMC and making it memoryless
def fit_vlmc(train_data, multi_source=False, thresh=2, prune_tree=True,
         prune_cutoff=-1, diff=True, full_tree=True, memoryless=True, prune_alpha=False):
    """Fits a VLMC. Generates a raw contexts tree and then prunes and otherwise modifies
    using the keyword arguments.  If multi_source is set to true, then the function assumes
    that train_data is a list of training sequences and will fit a vlmc for each element, adding
    them together before post processing.
    """
    if not multi_source:
        alphabet = set(train_data)
        mc = fit_context_tree(train_data, thresh, alphabet=alphabet)
    else:
        alphabet = set(train_data[0])
        if not np.all(set(td) == alphabet for td in train_data):
            raise Exception("All sequences must be over the same alphabet")
        all_trees = [fit_context_tree(t, thresh, alphabet=alphabet) for t in train_data]
        mc = reduce(add, all_trees)
    if prune_cutoff < 0:
        prune_cutoff = chi2.ppf(0.95, len(alphabet)-1)/2 #Cutoff for pruning.  Pick a better one later.
    if prune_tree:
        if not prune_alpha:
            prune(mc, delta_kl, prune_cutoff)
        else:
            prune_alpha_invest(mc, delta_kl, lambda x: chi2.ppf(1-x, len(alphabet)-1)/2)
    if full_tree:
        if memoryless:
            return make_fsm_closure(expand_nd(mc))
        return expand_nd(mc)
    if diff:
        return differentiate(mc)
    return mc

def cv_fit_vlmc(train_data, **kwargs):
    pass

def fit_context_tree(train_data, thresh, depth = 1,
                     filter_set = None, alphabet = None):
    if filter_set == None:
        filter_set = xrange(1, len(train_data))
    if alphabet == None:
        alphabet = set(train_data)
    counts = {s: 0 for s in alphabet}
    next_contexts = {s: [] for s in alphabet}
    for f in filter_set:
        counts[train_data[f]] += 1 if f != 0 else 0
        if f >= depth :
            next_contexts[train_data[f-depth]].append(f)
    children = {}
    for s in alphabet:
        #next_contexts = [i for i in filter_set if i >= depth and train_data[i-depth] == s]
        if len(next_contexts[s]) >= thresh:
            children[s] = fit_context_tree(train_data, thresh, depth+1, next_contexts[s], alphabet)
    return ContextTreeNode(counts, children)

class ContextTup(tuple):
#Allows sorting by context length in alpha investing
    def __lt__(self, other):
        return len(self) > len(other)


def prune_alpha_invest(mc, delta_f, alpha_to_delta, alpha=99):
    leaf_contexts = [ContextTup(x) for x in get_leaves(mc)]
    hq.heapify(leaf_contexts)
    parents = {}
    omega = 0
    while leaf_contexts and alpha > 0:
        child_ctxt = hq.heappop(leaf_contexts)
        par_ctxt = ContextTup(child_ctxt[1:])
        if par_ctxt not in parents:
            parents[par_ctxt] = get_node_by_context(mc, par_ctxt)
        cur_parent = parents[par_ctxt]
        cur_child = get_node_by_context(mc, child_ctxt)
        delta = delta_f(cur_parent, cur_child)
        bet_alpha = alpha/(len(child_ctxt) + 1)
        if delta < alpha_to_delta(bet_alpha):
            alpha -= bet_alpha /(1-bet_alpha)
            del cur_parent.children[child_ctxt[0]]
            if not cur_parent.children and len(par_ctxt) > 0:
                hq.heappush(leaf_contexts,ContextTup(par_ctxt))
        else :
            alpha += omega


def get_leaves_and_depths(mc, depth_dict=True):
    leaf_contexts = all_contexts(mc)
    leaves = [(len(c), get_node_by_context(mc, c)) for c in leaf_contexts]
    if depth_dict:
        leaves_by_depth = {}
        for depth, leaf in leaves:
            try:
                leaves_by_depth[depth].append(leaf)
            except:
                leaves_by_depth[depth] = [leaf]
        return leaves_by_depth
    return leaves

#def prune_alpha(mc, delta_f, cutoff_f, alpha):
#    for a in mc.children.keys():
#        child = mc.children[a]


def prune(mc, delta_f, cutoff):
    for a in mc.children.keys():
        child = mc.children[a]
        if prune(child, delta_f, cutoff): #If that child has no children after pruning
            if delta_f(mc, child) <= cutoff:
                del mc.children[a]
    if not mc.children:
        return True #It is a leaf
    return False #Internal node

def expand(mc, marginal=True):
    """Expands implicit terminal nodes to explicit ones.  Generally, this should
    not be used unless the tree has already been differentiated.
    Note that the way this expands the tree, the counts don't always represent
    observed counts anymore.
    The marginal argument is whether or not internal nodes should use the
    probabilities given by marginalizing over their children.  If false, all
    counts are set to zero."""
    if not mc.children:
        return copy_tree(mc)
    children = {}
    not_children = mc.alphabet.difference(mc.children)
    n_diff = len(not_children)
    for c in mc.children:
        children[c] = expand(mc.children[c], marginal)
    for c in not_children:
        new_counts = {s: count/n_diff for s, count in mc.counts.items()}
        children[c] = ContextTreeNode(new_counts, {})
    if marginal:
        marginal_counts = {a: sum([c.counts[a] for c in children.values()]) for a in mc.alphabet}
        marginal_probs = {a: v/sum(marginal_counts.values()) for a, v in marginal_counts.items()}
        for c in not_children:
            if not np.any(children[c].counts.values()):
                print "hi"
                children[c].update_prob(marginal_probs)
        return ContextTreeNode(marginal_counts, children)
    return ContextTreeNode({c : 0 for c in mc.alphabet}, children)

def expand_nd(mc):
    if not mc.children:
        return copy_tree(mc)
    children = {}
    not_children = mc.alphabet.difference(mc.children.keys())
    new_counts = mc.counts.copy()
    for label, child in mc.children.iteritems():
        for a in mc.alphabet:
            new_counts[a] -= child.counts[a]
        children[label] = expand_nd(child)
    new_tot = sum(new_counts.values())
    new_probs = {k: v/new_tot for k, v in new_counts.items()} if new_tot != 0 else mc.prob_dist.copy()
    for label in not_children:
        children[label] = ContextTreeNode({s:0 for s in mc.alphabet}, {}, new_probs)
    return ContextTreeNode(mc.counts.copy(), children)

def add_context(mc, new_context):
    """Adds a context with specified probabilities to an existing tree.
    Does not return anything"""
    if not new_context :
        return
    #tmp_mc = mc
    for c in reversed(new_context) :
        if c not in mc.children.keys() :
            #if counts == None:
            #    counts = temp_mc.counts
            mc.children[c] = ContextTreeNode(mc.counts, {}, mc.prob_dist)
        mc = mc.children[c]

def make_fsm_closure(mc):
    cp_mc = copy_tree(mc)
    contexts = all_contexts(mc)
    for c in contexts:
        for i in xrange(len(c)):
            add_context(cp_mc, c[0:i])
    return cp_mc


def delta_kl(parent, child):
    alphabet = [k for k in child.counts.keys() if child.counts[k] > 0]
    deltas = [child.counts[a] * log(child.prob(a) / max(parent.prob(a), 1e-7)) for a in alphabet]
    return sum(deltas)

def get_context_probs(mc, context):
    if not context:
        return mc.prob_dist#mc.counts, mc.totalcount
    return get_context_probs(mc.children[context[-1]], context[:-1])

def get_context(mc, sequence, internal=False):
    """Reads a string backwards to get the context the tree would use for it."""
    context = []
    for s in reversed(sequence):
        if s not in mc.children.keys():
            break
        context.append(s)
        mc = mc.children[s]
    context.reverse()
    if internal:
        return tuple(context)
    extra_contexts = all_contexts(mc)
    return [x + tuple(context) for x in extra_contexts]


def all_contexts(mc, path=[], implicit=False, interior=False):
    """Traverses the context tree of mc to give all possible contexts.
    implicit determines whether to print implicit states that are collapsed
    into non-full nodes.
    interior is whether to include nonterminal contexts
    """
    contexts = []
    if len(mc.children) != len(mc.alphabet):
        if implicit and len(mc.children) != 0:
            implicit_previous = ["[%s]" %"|".join([str(x) for x in mc.alphabet.difference(mc.children)])]
            contexts.append(implicit_previous + path)
        else:
            contexts.append(path)
    elif interior:
        contexts.append(path)
    child_contexts = [all_contexts(mc.children[a], [a]+path, implicit, interior) for a in mc.children.keys()]
    [contexts.extend(c) for c in child_contexts]
    return [tuple(c) for c in contexts]

def get_leaves(mc, path=[]):
    if not mc.children:
        return [tuple(path)]
    child_leaves = []
    for s, c in mc.children.items():
        child_leaves.extend(get_leaves(c, [s] + path))
    return child_leaves

def get_node_by_context(mc, context):
    node = mc
    for c in reversed(context):
        if c in node.children.keys():
            node = node.children[c]
        else:
            break
    return node

### Tree statistics
def entropy(mc):
    d = sum([c * log(c / mc.totalcount) for c in mc.counts.values() if c > 0]) if mc.totalcount > 0 else 0
    return d + sum([entropy(c) for c in mc.children.values()])

#def entropy2(mc1, mc2):
    
def holding_time_distribution(mc, sym):
    if sym not in mc.alphabet:
        raise Exception("Symbol not in alphabet")
    sym_t = (sym,)
    depth = mc.order
    seq_liks = [likelihood(mc, sym_t, depth), likelihood(mc, sym_t*2, depth)]
    denom = seq_liks[0] - seq_liks[1]
    num = lambda sl: sl[-1] - 2*sl[-2] + sl[-3]
    hts = []
    cur_node = mc
    d = 1
    while sym in cur_node.children.keys():
        cur_node = cur_node.children[sym]
        seq_liks.append(likelihood(mc, sym_t*(d+2), depth))
        hts.append(max(num(seq_liks) / denom, 0)) #Prevent roundoff from going below 0
        d += 1
    return (tuple(hts), 1-cur_node.prob_dist[sym])

def all_holding_time_distributions(mc):
    return {a: holding_time_distribution(mc, a) for a in mc.alphabet}


### Predictions
def predict_next(mc, context, pred_type="prob"):
    """Predicts the next observation given 'sequence' as context.
    If pred_type is not prob, will return the modal state"""
    child = mc
    for s in reversed(context):
        try:
            child = child.children[s]
        except:
            break
    #probs = {x: child.prob(x) for x in child.alphabet}
    if pred_type == "prob":
        return child.prob_dist
    maxkey = max(child.prob_dist.iteritems(), key=itemgetter(1))
    return maxkey[0]

def predict(mc, sequence, pred_type="prob"):
    return [predict_next(mc, sequence[:i], pred_type) for i in range(len(sequence))]

def likelihood(mc, seq, order_mc = None):
    if order_mc is None:
        order_mc = mc.order
    predictions = [predict_next(mc, seq[0:i])[s] for i, s in enumerate(seq)]
    return np.prod(predictions)

def context_marginal_probs(mc, interior=True):
    return {x:likelihood(mc, x) for x in all_contexts(mc, interior=interior)}

#This is not needed. I'm a moron.
def state_marginal_probs(mc, interior=False):
    """Computes the marginal probability of each state in the alphabet of the 
    VLMC"""
    context_mp = context_marginal_probs(mc, interior)
    state_mp = {s: 0 for s in mc.alphabet}
    for k, v in context_mp.iteritems():
        if len(k) == 0:
            print "Bad context?"
            print k
            continue
        state_mp[k[-1]] += v
    return state_mp

### Simulation
def simulate(mc, sim_len, discard=None):
    n_context_discard = 64
    order_mc = mc.order

    if discard is None or not isinstance(discard, (int, long)):
        discard = order_mc * n_context_discard
    sim = random.sample(mc.alphabet, 1) * (discard + sim_len)
    for i in xrange(order_mc, discard + sim_len):
        next_symbol_probs = predict_next(mc, sim[(i-order_mc):i])
        acc = 0
        r = random.random()
        for symbol, prob in next_symbol_probs.items():
            acc += prob
            if r < acc:
                sim[i] = symbol
                break
    return sim[discard:]

def q_np_head_geom_tail(head_c, tail_p, r):
    head_mass = head_c[-1]
    if r >= head_mass:
        tmp_q = 1 - (r - head_mass)/(1 - head_mass)
        return len(head_c) + int(np.ceil(np.log(tmp_q) / np.log(1 - tail_p)))
    return bisect.bisect_right(head_c, r) + 1

def simulate_holding_time(mc, sym, nsim=1):
    head_p, tail_p = holding_time_distribution(mc, sym)
    head_c = np.cumsum(head_p)
    return [q_np_head_geom_tail(head_c, tail_p, random.random()) for _ in xrange(nsim)]

def q_holding_time(mc, sym, p):
    head_p, tail_p = holding_time_distribution(mc, sym)
    head_c = np.cumsum(head_p)
    try:
        return [q_np_head_geom_tail(head_c, tail_p, q) for q in p]
    except TypeError:
        return q_np_head_geom_tail(head_c, tail_p, p)

###Tree manipulations
def add(mc1, mc2):
    """Adds two variable length markov chains"""
    if mc1.alphabet != mc2.alphabet:
        raise Exception("Cannot add together VLMCs with different alphabets.\n"
              + str(mc1.alphabet) + "\n" + str(mc2.alphabet))
    alphabet = mc1.alphabet
    if not mc1:
        return copy_tree(mc2)
    elif not mc2:
        return copy_tree(mc1)

    count_sums = {a:(mc1.counts[a] + mc2.counts[a]) for a in alphabet}
    children = {}
    k1, k2 = set(mc1.children.keys()), set(mc2.children.keys())
    for a in k1.intersection(k2):
        children[a] = add(mc1.children[a], mc2.children[a])
    for a in k1.difference(k2):
        children[a] = copy_tree(mc1.children[a])
    for a in k2.difference(k1):
        children[a] = copy_tree(mc2.children[a])

    return ContextTreeNode(count_sums, children)

def differentiate(mc, zero=False):
    if not mc.children:
        return copy_tree(mc)
    new_counts = mc.counts.copy()
    if zero or len(mc.alphabet) != len(mc.children):
    #If the node is full, then we will only end there if the string is too short
        for c in mc.children.values():
            new_counts = {a:(new_counts[a] - c.counts[a]) for a in mc.alphabet}
    return ContextTreeNode(new_counts, {a:differentiate(mc.children[a], zero)
                                           for a in mc.children.keys()})

def copy_tree(mc):
    newcounts = {k:v for k,v in mc.counts.iteritems()}
    newprobs = {k:v for k,v in mc.prob_dist.iteritems()}
    if not mc.children:
        return ContextTreeNode(newcounts, {}, newprobs)
    return ContextTreeNode(newcounts, {k:copy_tree(v) for k, v in mc.children.iteritems()}, newprobs)

###Graphical representations

def print_vlmc(mc, fmt='prob', rnd=4, **kwargs):
    print draw(mc, fmt, rnd, **kwargs)

def draw(mc, fmt='prob', rnd=4, level=0, label="ROOT", padding="", maxdepth=5):
    if level == 0:
        thisline = format_node(mc, label, fmt, rnd) + "\n"
    else:
        thisline =  padding[:-2] + "|_" + format_node(mc, label, fmt, rnd) + "\n"
    if level >= maxdepth:
        return thisline
    for index, k in enumerate(mc.children.keys()):
        if index < len(mc.children) - 1:
            lstart = "| "
        else:
            lstart = "  "
        thisline += \
        draw(mc.children[k], fmt, rnd, level+1, k, padding + lstart, maxdepth)
    return thisline

def format_node(node, nodelabel, fmt, rnd=3):
    if fmt != 'prob':
        kidcounts = ["%s: %d" %(str(k), v) for k, v in node.counts.iteritems()]
        return str(nodelabel) + "(%s | %d)" %(", ".join(kidcounts), node.totalcount)
    fmtstr = "%%s: %%.%df" %(rnd)
    return str(nodelabel) + "(%s)" %(", ".join([fmtstr %(str(k), round(v, rnd)) for k, v in node.prob_dist.iteritems()]))


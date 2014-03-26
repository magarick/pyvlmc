from __future__ import division
from contexttree import ContextTreeNode
from pyvlmc.vlmc import all_contexts, vlmc, copy_tree

def memoryless_vlmc(train_data, marginal=True, *args, **kwargs):
    return make_fsm_closure(expand(vlmc(train_data, *args, **kwargs)))

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
    for c in not_children:
        new_counts = {s: count/n_diff for s, count in mc.counts.items()}
        children[c] = ContextTreeNode(new_counts, {})
    for c in mc.children:
        children[c] = expand(mc.children[c], marginal)
    if marginal:
        marginal_counts = {a: sum([c.counts[a] for c in children.values()]) for a in mc.alphabet}
        return ContextTreeNode(marginal_counts, children)
    return ContextTreeNode({c : 0 for c in mc.alphabet}, children)

def add_context_to_vlmc(mc, new_context, counts=None):
    """Adds a context with specified probabilities to an existing tree.
    Does not return anything"""
    if not new_context :
        return
    temp_mc = mc
    for c in reversed(new_context) :
        if c not in temp_mc.children.keys() :
            if counts == None:
                counts = temp_mc.counts
            temp_mc.children[c] = ContextTreeNode(counts, {})
        temp_mc = temp_mc.children[c]

def make_fsm_closure(mc):
    cp_mc = copy_tree(mc)
    contexts = all_contexts(mc)
    add_context = lambda x : add_context_to_vlmc(cp_mc, x)
    for c in contexts:
        map(add_context, [c[0:i] for i in xrange(len(c))])
    return cp_mc

#Not necessary now
#def find_context_violations(mc):
#    contexts = all_contexts(mc)
#    def tup_endswith(t1, t2):
#        return all([v1 == v2 for v1, v2 in zip(reversed(t1), reversed(t2))])
#    violation_list = [(c, c2) for c2 in contexts for c in contexts
#            if len(c2) > len(c) + 1 and tup_endswith(c, c2[:-1])]
#    violation_dict = {}
#    for c, c2 in violation_list:
#        try:
#            violation_dict[c].append(c2)
#        except:
#            violation_dict[c] = [c2]
#    return violation_dict
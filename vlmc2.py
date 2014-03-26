# -*- coding: utf-8 -*-

from __future__ import division
from math import log
from operator import itemgetter
from scipy.stats import chi2
import numpy as np
import random
from contexttree import ContextTreeNode
import bisect
import heapq as hq

def fit_dual_tree(in_seq, out_seq, thresh, depth = 1,
                     filter_set = None, alphabet = None):
    if len(in_seq) != len(out_seq):
        raise Exception("Predictor and output sequences not same length.")
    if filter_set == None:
        filter_set = xrange(1, len(in_seq))
    if alphabet == None:
        alphabet = {'out': set(out_seq), 'in': set(in_seq)}
    counts = {s: 0 for s in alphabet['out']}
    next_contexts = {s: [] for s in alphabet['in']}
    for f in filter_set:
        counts[out_seq[f]] += 1 if f != 0 else 0
        if f >= depth :
            next_contexts[in_seq[f-depth]].append(f)
    children = {}
    for s in alphabet:
        if len(next_contexts[s]) >= thresh:
            children[s] = fit_dual_tree(in_seq, out_seq, thresh, depth+1,
                                             next_contexts[s], alphabet)
    return ContextTreeNode(counts, children)
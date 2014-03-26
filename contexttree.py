
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 13:42:16 2013

@author: josh
"""
from __future__ import division
from collections import namedtuple
import json

ContextTreeNodeT = namedtuple('ContextTreeNodeT',
                              ['counts', 'totalcount', 'children', 'alphabet', 'prob_dist'])


def ContextTreeNode(counts, children, prob_dist=None):
    totalcount = sum(counts.itervalues())
    if prob_dist is None:
        if totalcount != 0:
            prob_dist = {s: c/totalcount for s, c in counts.iteritems()}
        else:
            prob_dist = {s: 0 for s in counts.iterkeys()}
    alphabet = frozenset(counts.keys())
    return ContextTreeNodeT(counts, totalcount, children, alphabet, prob_dist)

###Class methods to be tacked on after
def ct_prob(self, a):
    if self.prob_dist is None:
        return self.counts[a] / self.totalcount if self.totalcount > 0 else 0
    return self.prob_dist[a]


def update_next_symbol_probs(self, posterior):
    if posterior.keys() != self.prob_dist.keys():
        raise Exception("New and old next state distribution must be defined on same symbols.")
    else:
        for k in self.prob_dist.iterkeys():
            self.prob_dist[k] = posterior[k]


def to_dict(self):
    root = {'counts': self.counts, 'prob_dist': self.prob_dist}
    if not self.children:
        root['children'] = {}
        return root
    root['children'] = {k: v.to_dict() for k, v in self.children.iteritems()}
    return root


def to_json(self):
    return json.dumps(self.to_dict())


def from_json(json_str):
    return from_dict(json.loads(json_str))


def from_dict(c_dict):
    if not 'counts' in c_dict and not 'prob_dist' in c_dict:
        raise Exception("Tried to form a ContextTreeNode with no counts or probabilities!")
    try:
        cnt = c_dict['counts']
    except KeyError:
        cnt = {k: 0 for k in c_dict['prob_dist'].keys()}
    try:
        pdist = c_dict['prob_dist']
    except KeyError:
        pdist = None
    if c_dict['children'] == {} or 'children' not in c_dict:
        return ContextTreeNode(cnt, {}, pdist)
    children = {k: from_dict(v) for k, v in c_dict['children'].iteritems()}
    return ContextTreeNode(cnt,  children, pdist)


@property
def order(mc):
    if len(mc.children) == 0:
        return 0
    return 1 + max([x.order for x in mc.children.values()])


@property
def num_leaves(mc):
    if not mc.children:
        return 1
    return sum([x.num_leaves for x in mc.children.values()])


@property
def num_terminal_nodes(mc):
    if not mc.children:
        return 1
    elif len(mc.children) != len(mc.alphabet):
        return 1 + sum([x.num_terminal_nodes for x in mc.children.values()])
    return sum([x.num_terminal_nodes for x in mc.children.values()])


@property
def num_nodes(mc):
    if not mc.children:
        return 1
    return 1 + sum([x.num_nodes for x in mc.children.values()])

#@property
#def totalcount(self):
#    return sum(self.counts.itervalues())

ContextTreeNodeT.prob = ct_prob
ContextTreeNodeT.update_prob = update_next_symbol_probs
ContextTreeNodeT.order = order
ContextTreeNodeT.num_leaves = num_leaves
ContextTreeNodeT.num_terminal_nodes = num_terminal_nodes
ContextTreeNodeT.num_nodes = num_nodes
ContextTreeNodeT.to_dict = to_dict
ContextTreeNodeT.to_json = to_json

#ContextTreeNodeT.totalcount = totalcount
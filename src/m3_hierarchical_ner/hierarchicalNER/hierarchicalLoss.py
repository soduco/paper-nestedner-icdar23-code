import numpy as np
import torch
from typing import List
from nltk.tree import Tree

def get_label(node):
    """
    Source : https://github.com/fiveai/making-better-mistakes/blob/f0bb7d0a7c4a9c7c9517045a4be4446e9baf824f/better_mistakes/trees.py
    :param node:
    :return:
    """
    if isinstance(node, Tree):
        return node.label()
    else:
        return node

class HierarchicalLLLoss(torch.nn.Module):
    """
    Hierachical log likelihood loss.
    The weights must be implemented as a nltk.tree object and each node must
    be a float which corresponds to the weight associated with the edge going
    from that node to its parent. The value at the origin is not used and the
    shapre of the weight tree must be the same as the associated hierarchy.
    The input is a flat probability vector in which each entry corresponds to
    a leaf node of the tree. We use alphabetical ordering on the leaf nodes
    labels, which corresponds to the 'normal' imagenet ordering.
    Source : https://github.com/fiveai/making-better-mistakes/blob/f0bb7d0a7c4a9c7c9517045a4be4446e9baf824f/better_mistakes/model/losses.py
    Args:
        hierarchy: The hierarchy used to define the loss.
        classes: A list of classes defining the order of the leaf nodes.
        weights: The weights as a tree of similar shape as hierarchy.
    """

    def __init__(self, 
                 #HierarchicalLLLoss Parameters
                 hierarchy: Tree, 
                 classes: List[str], 
                 weights: Tree,
                 ignore_index : int=-100):
        super(HierarchicalLLLoss, self).__init__()
        
        self.ignore_index = ignore_index
        
        #Vérifie que les structures des deux arbres sont identiques
        assert hierarchy.treepositions() == weights.treepositions()

        # The tree positions of all the leaves
        positions_leaves = {get_label(hierarchy[p]): p for p in hierarchy.treepositions("leaves")}
        print(f"Tree positions of all the leaves{positions_leaves} #1")
        num_classes = len(positions_leaves)
        print(f"Num of classes : {num_classes}")
        # We use classes in the given order
        positions_leaves = [positions_leaves[c] for c in classes]
        # the tree positions of all the edges (we use the bottom node position)
        positions_edges = hierarchy.treepositions()[1:]  # the first one is the origin
        # map from position tuples to leaf/edge indices
        index_map_leaves = {positions_leaves[i]: i for i in range(len(positions_leaves))}
        index_map_edges = {positions_edges[i]: i for i in range(len(positions_edges))}
        # edge indices corresponding to the path from each index to the root
        edges_from_leaf = [[index_map_edges[position[:i]] for i in range(len(position), 0, -1)] for position in positions_leaves]
        # get max size for the number of edges to the root
        num_edges = max([len(p) for p in edges_from_leaf])
        
        # helper that returns all leaf positions from another position wrt to the original position
        def get_leaf_positions(position):
            node = hierarchy[position]
            if isinstance(node, Tree):
                return node.treepositions("leaves")
            else:
                return [()]

        # indices of all leaf nodes for each edge index
        leaf_indices = [[index_map_leaves[position + leaf] for leaf in get_leaf_positions(position)] for position in positions_edges]
        # save all relevant information as pytorch tensors for computing the loss on the gpu
        self.onehot_den = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges], device="cuda:0"), requires_grad=False)
        self.onehot_num = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges], device="cuda:0"), requires_grad=False)
        self.weights = torch.nn.Parameter(torch.zeros([num_classes, num_edges], device="cuda:0"), requires_grad=False)
        
        # one hot encoding of the numerators and denominators and store weights
        for i in range(num_classes):
            for j, k in enumerate(edges_from_leaf[i]):
                self.onehot_num[i, leaf_indices[k], j] = 1.0
                self.weights[i, j] = get_label(weights[positions_edges[k]])
            for j, k in enumerate(edges_from_leaf[i][1:]):
                self.onehot_den[i, leaf_indices[k], j] = 1.0
            self.onehot_den[i, :, j + 1] = 1.0  # the last denominator is the sum of all leaves
        
    def forward(self, inputs, target):
        """
        Forward pass, computing the loss.
        Args:
            inputs: Class _probabilities_ ordered as the input hierarchy.
            target: The index of the ground truth class.
        """

        # We start by filtering items which are marked so.
        ignore_mask = target != self.ignore_index
        target_clean = target[ignore_mask]
        inputs_clean = inputs[ignore_mask]

        # Then we get back to the original "Make better mistakes" code
        # ---------
        # add a sweet dimension to inputs
        inputs_clean = torch.unsqueeze(inputs_clean, 1) #Vecteur en dimension d'un tensor https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        # sum of probabilities for numerators
        num = torch.squeeze(torch.bmm(inputs_clean, self.onehot_num[target_clean]))
        # print(f"Num {num}, num.shape {num.shape}")
        # sum of probabilities for denominators
        den = torch.squeeze(torch.bmm(inputs_clean, self.onehot_den[target_clean])) # Idem
        # compute the neg logs for non zero numerators and store in there
        # We assume here that all(den != 0), which means than the tree branches must all have the same length
        idx = (num != 0)
        num[idx] = -torch.log(num[idx] / den[idx])
        # weighted sum of all logs for each path (we flip because it is numerically more stable)
        num = torch.sum(torch.flip(self.weights[target_clean] * num, dims=[1]), dim=1)
        # return sum of losses / batch size
        return torch.mean(num) #Valeur de la loss par batch. Il semble que ce calcul par batch soit fait par le trainer aussi.
        #Ce que retourne la Cross entropy ets différent
        #https://programmer.group/calculation-of-cross_entropy-loss-function-of-torch-including-python-code.html

class HierarchicalCrossEntropyLoss(HierarchicalLLLoss):
    """
    Combines softmax with HierachicalNLLLoss. Note that the softmax is flat.
    Source : https://github.com/fiveai/making-better-mistakes/blob/f0bb7d0a7c4a9c7c9517045a4be4446e9baf824f/better_mistakes/model/losses.py
    """

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree, ignore_index : int=-100):
        super(HierarchicalCrossEntropyLoss, self).__init__(hierarchy, classes, weights, ignore_index)

    def forward(self, inputs, index):
        return super(HierarchicalCrossEntropyLoss, self).forward(torch.nn.functional.softmax(inputs, 1), index)
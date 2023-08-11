from nltk.tree import Tree

############## IO Trees

#1. Labels tree
o_io = Tree('O', ['O+O'])
names_io = Tree('I-geogName', ['I-geogName+O','I-geogName+i_geogFeat','I-geogName+i_name'])
Ltree = Tree('EN', [o_io,names_io]) #EN = tree's root

#2. Weights tree
Wo_io = Tree(1., [1.])
Wnames_io = Tree(1., [1.,1.,1.])

Wtree = Tree(-100., [Wo_io, Wnames_io]) #-100 = tree's root
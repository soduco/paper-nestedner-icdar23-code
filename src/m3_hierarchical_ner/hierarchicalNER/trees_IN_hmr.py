from nltk.tree import Tree

############## IO Trees

#1. Labels tree
Ltree = Tree.fromstring("""(EN (O (O (O (O O+O+O+O)))) 
                (I-geogFeat (O (O (O I-geogFeat+O+O+O))))
                (I-geogName (O (O (O I-geogName+O+O+O))) (I-geogFeat (O (O I-geogName+i_geogFeat+O+O))) (I-name (O (O I-geogName+i_name+O+O))) (I-geogName (O (O I-geogName+i_geogName+O+O)) (I-name (O I-geogName+i_geogName+i_name+O)) (I-geogFeat (O I-geogName+i_geogName+i_geogFeat+O)) (I-geogName (O I-geogName+i_geogName+i_geogName+O) (I-geogFeat I-geogName+i_geogName+i_geogName+i_geogFeat) (I-name I-geogName+i_geogName+i_geogName+i_name)))
                ) 
                )
                """)

#2. Weights tree
Wtree = Tree.fromstring("""(-100. (1. (1. (1. (1. 1.)))) 
                (1. (1. (1. (1. 1.))))
                (1. (1. (1. (1. 1.))) (1. (1. (1. 1.))) (1. (1. (1. 1.))) (1. (1. (1. 1.)) (1. (1. 1.)) (1. (1. 1.)) (1. (1. 1.) (1. 1.) (1. 1.)))
                ) 
                )
                """)


Wo_io = Tree(1., [Tree(1., [Tree(1., [Tree(1., [1.])])])])
Wnames_io = Tree(1., [1.,1.,1.])

Wtree = Tree(-100., [Wo_io, Wnames_io]) #-100 = tree's root
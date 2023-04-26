from nltk.tree import Tree

############## IO Trees

#1. Labels tree
o_io = Tree('O', ['O+O'])
per_io = Tree('I-PER', ['I-PER+O','I-PER+i_TITREH'])
desc_io = Tree('I-DESC', ['I-DESC+O','I-DESC+i_ACT','I-DESC+i_TITREP'])
act_io = Tree('I-ACT', ['I-ACT+O'])
spat_io = Tree('I-SPAT', ['I-SPAT+O','I-SPAT+i_LOC','I-SPAT+i_CARDINAL','I-SPAT+i_FT'])
titre_io = Tree('I-TITRE', ['I-TITRE+O'])

Ltree = Tree('EN', [o_io,per_io, act_io, desc_io, spat_io, titre_io]) #EN = tree's root

#2. Weights tree
Wo_io = Tree(1., [1.])
Wper_io = Tree(1., [1.,1.])
Wdesc_io = Tree(1., [1.,1.,1.])
Wact_io = Tree(1., [1.])
Wspat_io = Tree(1., [1.,1.,1.,1.])
Wtitre_io = Tree(1., [1.])

Wtree = Tree(-100., [Wo_io, Wper_io, Wact_io, Wdesc_io, Wspat_io, Wtitre_io]) #-100 = tree's root

############## IOB2 Trees (first version)

o_iob2 = Tree('O', [Tree('O+O',['O+O'])])
per_iob2 = Tree('PER', [Tree('PER+O',['I-b_PER+O','I-i_PER+O']),Tree('PER+TITREH',['I-b_PER+b_TITREH','I-i_PER+b_TITREH','I-i_PER+i_TITREH'])])
desc_iob2 = Tree('DESC', [Tree('DESC+O',['I-b_DESC+O','I-i_DESC+O']),Tree('DESC+ACT',['I-b_DESC+b_ACT','I-i_DESC+b_ACT','I-i_DESC+i_ACT']),Tree('DESC+TITREP',['I-b_DESC+b_TITREP','I-i_DESC+b_TITREP','I-i_DESC+i_TITREP'])])
act_iob2 = Tree('ACT', [Tree('ACT+O',['I-b_ACT+O','I-i_ACT+O'])])
spat_iob2 = Tree('SPAT', [Tree('SPAT+O',['I-b_SPAT+O','I-i_SPAT+O']),Tree('SPAT+LOC',['I-b_SPAT+b_LOC','I-i_SPAT+b_LOC','I-i_SPAT+i_LOC']),Tree('SPAT+CARDINAL',['I-b_SPAT+b_CARDINAL','I-i_SPAT+b_CARDINAL','I-i_SPAT+i_CARDINAL']),Tree('SPAT+FT',['I-b_SPAT+b_FT','I-i_SPAT+b_FT','I-i_SPAT+i_FT'])])
titre_iob2 = Tree('TITRE', [Tree('TITRE+O', ['I-b_TITRE+O','I-i_TITRE+O'])])

Ltree_iob2 = Tree('EN', [o_iob2,per_iob2, act_iob2, desc_iob2, spat_iob2, titre_iob2]) #EN = tree's root

#2. Weights tree
Wo_iob2 = Tree(1., [Tree(1.,[1.])])
Wper_iob2 = Tree(1., [Tree(1.,[1.,1.]),Tree(1.,[1.,1.,1.])])
Wdesc_iob2 = Tree(1., [Tree(1.,[1.,1.]),Tree(1.,[1.,1.,1.]),Tree(1.,[1.,1.,1.])])
Wact_iob2 = Tree(1., [Tree(1.,[1.,1.])])
Wspat_iob2 = Tree(1., [Tree(1.,[1.,1.]),Tree(1.,[1.,1.,1.]),Tree(1.,[1.,1.,1.]),Tree(1.,[1.,1.,1.])])
Wtitre_iob2 = Tree(1., [Tree(1.,[1.,1.])])

Wtree_iob2 = Tree(-100., [Wo_iob2, Wper_iob2, Wact_iob2, Wdesc_iob2, Wspat_iob2, Wtitre_iob2]) #-100 = tree's root
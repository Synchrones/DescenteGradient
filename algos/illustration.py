from Reseau_objet import *

reseau = Reseau(nb_entrees=2, couches=[(1, Relu, derive_Relu), (2, Relu, derive_Relu)])
# reseau.couches = [[Neurone1,1], [Neurone2,1, Neurone2,2]]
# Neurone1,1.poids = [poids1, poids2, poids3] car 2 entrées sur la couche précédente

neurone = Neurone(poids=[0.2, 0.3, 0.1], fonction_activation=Relu, derive_fonction_activation=Relu)


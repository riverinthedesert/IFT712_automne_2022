## Sujet
Programmation des algorithmes de régression linéaire et non linéaire polynomial vus aux chapitres 1 et 3 ainsi que la recherche d’hyperparamètre « k-fold cross-validation » vu au chapitre 1.  
Les algorithmes sont implémentés via la classe Regression du fichier solution_regression.py.

## Equipe
Equipe 7  :
  - caiy2401 - CAI, Yunfan
  - gaye1902 - ElHadji Habib Gaye
  - aity1101 - Yoann Ait Ichou

## Remarque
 - IFT712_Devoir1_Theorique_AIT_ICHOU_CAI_GAYE.pdf correspond aux exercice 1 à 4 au format pdf 
 - solution_regression.py et regression.py correspondent à l'exercice 5 de programmation en python

## Url gitlab
https://github.com/riverinthedesert/IFT712_automne_2022

Veuiller vérifier sur le lien github s'il manque quelque chose dans notre dépot.

## Informations complémentaires
Lors de l'exécution de l'algorithme, pour un petit jeu d'entrainement, la prédiction obtenue a plus de chance d'être le résultat d'un sur-apprentissage.
L'environnement de test est l'IDE spyder sous python 3.9.
## Exemple de commandes 

Petit jeu de donnée avec recherche d'hyperparamètre:
```
python3 regression.py 1 tan 20 20 0.3 -1 0.001
```

Grand jeu de donnée sans recherche d'hyperparamètre:
```
python3 regression.py 1 tan 100 100 0.3 10 0.001
```
# -*- coding: utf-8 -*-

#####
# Yoann AIT ICHOU (Matricule)
###

import numpy as np
import random
from sklearn import linear_model


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """

        phi_x = []
        if np.isscalar(x) :
            # x est un scalaire
            for n in range(self.M + 1) :
                phi_x.append(x ** n)
            phi_x = np.array(phi_x)
        else :
            # x est un vecteur de N scalaires
            for i in x :
                phi_x.append([i ** n for n in range(self.M + 1)])
            phi_x = np.array(phi_x)
            
        return phi_x

    def recherche_hyperparametre(self, X, t):
        """
        Trouver la meilleure valeur pour l'hyper-parametre self.M (pour un lambda fixe donné en entrée).

        Option 1
        Validation croisée de type "k-fold" avec k=10. La méthode array_split de numpy peut être utlisée 
        pour diviser les données en "k" parties. Si le nombre de données en entrée N est plus petit que "k", 
        k devient égal à N. Il est important de mélanger les données ("shuffle") avant de les sous-diviser
        en "k" parties.

        Option 2
        Sous-échantillonage aléatoire avec ratio 80:20 pour Dtrain et Dvalid, avec un nombre de répétition k=10.

        Note: 

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        M_max = 11
        k = 10
        shuffled = []
        shuffled_X = []
        shuffled_t = []

        for i in range(len(X)):
            shuffled.append([X[i], t[i]])
        random.shuffle(shuffled)
        for i in range(len(X)):
            shuffled_X.append(shuffled[i][0])
            shuffled_t.append(shuffled[i][1])

        shuffled_X = np.array(shuffled_X)
        shuffled_t = np.array(shuffled_t)

        k_folds_X = np.array_split(shuffled_X, k)
        k_folds_t = np.array_split(shuffled_t, k)

        liste_erreur = []

        for m in range(1, M_max + 1):
            regression = Regression(self.lamb, m)
            erreur_moyenne = 0
            for i in range(k):
                validation_X = k_folds_X[i]
                validation_t = k_folds_t[i]
                entrainement_X = None
                entrainement_t = None

                for j in range(k):
                    if j != i:
                        if entrainement_X is None:
                            entrainement_X = k_folds_X[j]
                            entrainement_t = k_folds_t[j]
                        else:
                            entrainement_X = np.concatenate(
                                (entrainement_X, k_folds_X[j]))
                            entrainement_t = np.concatenate(
                                (entrainement_t, k_folds_t[j]))
                regression.entrainement(entrainement_X, entrainement_t)
                erreur = 0

                for j in range(len(validation_X)):
                    prediction = regression.prediction(validation_X[j])
                    erreur = erreur + \
                        regression.erreur(validation_t[j], prediction)

                erreur_moyenne = erreur_moyenne + erreur
            erreur_moyenne = erreur_moyenne/k
            liste_erreur.append(erreur_moyenne)

        erreur_min = liste_erreur[0]
        indice_erreur_min = 0

        for i in range(len(liste_erreur)):
            if(liste_erreur[i] < erreur_min):
                erreur_min = liste_erreur[i]
                indice_erreur_min = i

        self.M = indice_erreur_min + 1
        print("L'hyperparamètre retenu est M = "+str(self.M))

    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.

        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de 
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)

        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)
        
        phi_x = self.fonction_base_polynomiale(X)

        if using_sklearn:
            # la classe "Ridge"
            reg = linear_model.Ridge(alpha=self.lamb)
            reg.fit(phi_x, t)
            self.w = reg.coef_
            self.w[0] = reg.intercept_
        else:
            # procedure de resolution de systeme d'equations lineaires
            self.w = np.linalg.solve(self.lamb * np.identity(len(phi_x.T)) + phi_x.T.dot(phi_x), phi_x.T.dot(t))

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """

        phi_x = self.fonction_base_polynomiale(x)
        return np.dot(self.w.T, phi_x.T)

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        return (prediction - t) ** 2

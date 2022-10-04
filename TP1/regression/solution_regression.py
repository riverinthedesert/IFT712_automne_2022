# -*- coding: utf-8 -*-

#####
#   - caiy1401 - CAI, Yunfan
###

import numpy as np
import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd


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
        # AJOUTER CODE ICI
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
        # AJOUTER CODE ICI
        
        train = pd.DataFrame(X)
        test = pd.DataFrame(t)
        
        best_accu = 1
        best_params = None
        score = 0
        
        M_choices = [1, 2, 3, 5, 10, 20]
        nbK = 10
        
        for x in range(0, nbK):
            
            # split 20% of train et valid data pour la evaluation
            X_train, X_test, y_train, y_test = train_test_split(train, test,
                test_size=0.2, shuffle = True, random_state = 8)
            
            
            # # meme pour la validation set
            # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
            #     test_size=0.2, random_state= 8)
            
            for M_param in M_choices:
                self.M = M_param
                self.entrainement(X_train, y_train)
                prdiction = self.prediction(X_test)
                score = self.erreur(y_test, prdiction)
            
                if score < best_accu:
                    best_accu = score
                    best_params = M_param
            
        
        self.M = best_params

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
        
        print(X.shape)
        print(t.shape)
        
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

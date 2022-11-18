# -*- coding: utf-8 -*-

#####
# Vos Noms (Vos Matricules) .~= À MODIFIER =~.
###

import numpy as np
import matplotlib.pyplot as plt


class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, olynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None
        
    
    def kernel(self, x1, x2):
        """
        Calcule le noyau entre deux vecteurs en fonction de 'self.noyau'.
        """
        if self.noyau == "rbf":
            return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.sigma_square))
        elif self.noyau == "lineaire":
            return x1.dot(x2.T)
        elif self.noyau == "polynomial":
            return np.power(self.c + x1.dot(x2.T), self.M)
        elif self.noyau == "sigmoidal":
            return np.tanh((self.b * x1.dot(x2.T) + self.d))

        

    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """
        
        self.x_train = x_train
        K = np.array([[self.kernel(x_train[i], x_train[j]) for j in range(len(x_train))] for i in range(len(x_train))])
        self.a = np.linalg.inv(K + self.lamb * np.identity(len(K))).dot(t_train)
        
    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        k_x = np.array([self.kernel(x1, x) for x1 in self.x_train])
        y_x = np.dot(k_x, self.a)
        return int(y_x >= 0.5)

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        return (prediction - t) ** 2
    
    
    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=10 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        # Define hyperparameters values search grid (achived by sklearn)
        # param = {}
        # if self.noyau == "rbf":
        #     param.update({"sigma_square": list(np.linspace(0.000000001, 2., 15))})
        # elif self.noyau == "polynomial":
        #     param.update({"M": list(np.arange(2, 6, 1)),
        #                   "c": list(np.linspace(0, 5, 10))})
        # elif self.noyau == "sigmoidal":
        #     param.update({"b": list(np.linspace(0.00001, 0.01, 10)),
        #                   "d": list(np.linspace(0.00001, 0.01, 10))})
        # param.update({"lamb": list(np.linspace(0.000001, 2., 10))})
        # grilleRecherche = list(ParameterGrid(param))
        
        #Define hyperparameters values search grid (achived by numpy)
        grilleRecherche = list()
        if self.noyau == "rbf":
            for grid_lamb in list(np.linspace(0.000001, 2., 10)):
                for grid_sigma_square in list(np.linspace(0.000000001, 2., 15)):
                    grilleRecherche.append({"sigma_square": grid_sigma_square ,"lamb": grid_lamb})
        elif self.noyau == "polynomial":
            for grid_lamb in list(np.linspace(0.000001, 2., 10)):
                for grid_M in list(np.arange(2, 6, 1)):
                    for grid_c in list(np.linspace(0, 5, 10)):
                                        grilleRecherche.append({"M": grid_M,
                                                                "c": grid_c,
                                                                "lamb": grid_lamb})
        elif self.noyau == "sigmoidal":
            for grid_lamb in list(np.linspace(0.000001, 2., 10)):
                for grid_b in list(np.linspace(0.00001, 0.01, 10)):
                    for grid_d in list(np.linspace(0.00001, 0.01, 10)):
                                        grilleRecherche.append({"b": grid_b,
                                                                "d": grid_d,
                                                                "lamb": grid_lamb})

        print(np.shape(grilleRecherche))


        # Cross validation
        folds=10
        hyper_mean_error = {}
        size_folds = len(x_tab) // 10
        for hyper in range(len(grilleRecherche)):
            # Setup hyper parameters
            self.lamb = grilleRecherche[hyper]["lamb"]
            if self.noyau == "rbf":
                self.sigma_square = grilleRecherche[hyper]["sigma_square"]
            elif self.noyau == "polynomial":
                self.M = grilleRecherche[hyper]["M"]
                self.c = grilleRecherche[hyper]["c"]
            elif self.noyau == "sigmoidal":
                self.b = grilleRecherche[hyper]["b"]
                self.d = grilleRecherche[hyper]["d"]

            fold_mean_error = 0
            for fold in range(folds):
                # Split the train and test data in correlation to the actual fold
                test_start = size_folds * fold
                test_end = size_folds * (fold + 1)
                test_X = x_tab[test_start: test_end]
                test_t = t_tab[test_start: test_end]
                train_X = np.concatenate((x_tab[:test_start], x_tab[test_end:]))
                train_t = np.concatenate((t_tab[:test_start], t_tab[test_end:]))
                # Train and test the hyperparameter
                self.entrainement(train_X, train_t)
                pred_t = np.array([self.prediction(elem) for elem in test_X])
                fold_mean_error += np.mean(self.erreur(test_t, pred_t))

            # Compute the mean error for the m value
            hyper_mean_error[hyper] = fold_mean_error / folds

        # Re train with best hyper parameter
        hyper = min(hyper_mean_error, key=hyper_mean_error.get)
        print("Best hyperparameters:", grilleRecherche[hyper])
        self.lamb = grilleRecherche[hyper]["lamb"]
        if self.noyau == "rbf":
            self.sigma_square = grilleRecherche[hyper]["sigma_square"]
        elif self.noyau == "polynomial":
            self.M = grilleRecherche[hyper]["M"]
            self.c = grilleRecherche[hyper]["c"]
        elif self.noyau == "sigmoidal":
            self.b = grilleRecherche[hyper]["b"]
            self.d = grilleRecherche[hyper]["d"]
        self.entrainement(train_X, train_t)

    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()

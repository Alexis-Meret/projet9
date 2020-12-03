# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:44:16 2020

@author: User
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot  as plt
from scipy.optimize import curve_fit

"""fonction 1 : etalon --> donne a,b de T=a*val_pixe +b"""
im=Image.open("tige4.jpg")
tab=np.array(im,dtype="float")
def etalon (tab,Tmin,Tmax):
    etalon=tab[30:208,310]
    n,m=np.shape((etalon))
    moy=np.zeros((n))
    for i in range (n):
        moy[i]=np.mean(etalon[i,:])
    a=(Tmax-Tmin)/(moy[0]-moy[n-1])
    b=Tmax-a*moy[0]
    return a,b


"""fonction2 : extraction ---> renvoie un sous tableau des val moy des pixels associé à un objet fin vertical"""
def extraction (tab,ligne_debut,ligne_fin,colonne_debut,colonne_fin):
    sous_tab=tab[ligne_debut:ligne_fin,colonne_debut:colonne_fin]
    n,m,l=np.shape(sous_tab)
    tab_moy=np.zeros((n),dtype="float")
    for i in range(n):
        tab_moy[i]=np.mean(sous_tab[i])
    return tab_moy


"""fonction3 : conversion --->conversion d'une valeur moyenne de pixel en temperature"""
def convesrion_T (tab_moy,a,b):
    n=len(tab_moy)
    tempe=np.zeros(n,dtype="float")
    for i in range(n):
        tempe[i] = a*tab_moy[i]+b
    return tempe

"""fonction4 : conversion --->conversion pixel/distance"""
def convesrion_z (tab_moy,L):
    n=len(tab_moy)
    z=np.zeros(n)
    for i in range(n):
        z[i]=i*L/(n-1)
    return z

"""fonction 5 : tracé de T(z)"""

def Regression_MC(X,Y,u_Y):
    def f(X,a,b,tau):#on introduit ici tous les paramètres d'ajustement de la fonction de régression
        return a+b*np.exp(-X/tau)#on donne ici l'expression de la fonction de régression
    N=5#nombre de fois que la Methode de Monté Carlo est réalisée
    tab_a=np.zeros((N))
    tab_b=np.zeros((N))
    tab_tau=np.zeros((N))
    for i in range (N):#Méthode de Monté Carlo
        Y_new=np.random.normal(Y,u_Y)
        popt,pcov=curve_fit(f,X,Y_new,sigma=u_Y)#détermination des paramètres d'ajustement par la méthode des moindres carrés
        a=popt[0]
        b=popt[1]
        tau=popt[2]
        tab_a[i]=a
        tab_b[i]=b
        tab_tau[i]=tau
    a=np.mean(tab_a)#valeur moyenne des diiférents paramètres d'ajustement
    b=np.mean(tab_b)
    tau=np.mean(tab_tau)
    u_a=np.std(tab_a)#incertitude-type des différents paramètres d'ajustement.
    u_b=np.std(tab_b)
    u_tau=np.std(tab_tau)
    x=np.linspace(np.min(X),np.max(X),1000)
    y=f(x,a,b,tau)#expression de la droite de régression
    plt.errorbar(X,Y,yerr=u_Y,linestyle="",marker='o',label="points expérimentaux")#points exp + barres d'erreur
    plt.plot(x,y,label="T=a+b*np.exp(-z/$\delta$) \n\
    a = ({0:.1f} +/- {1:.1f}) °C\n\
    b = ({2:.1f} +/- {3:.1f}) °C\n\
    ($\delta$ = {4:.3f} +/- {5:.3f})m".format(a,u_a,b,u_b,tau,u_tau))
    plt.grid(True)
    plt.xlabel("z(m)")
    plt.ylabel("temperature(°C)")
    plt.title("regression")
    plt.legend()
    plt.show()


tab_moy = extraction(tab,10,238,247,248)
a,b=etalon(tab,18.4,60.2)
tab_T = convesrion_T(tab_moy,a,b)
tab_z=convesrion_z(tab_moy,0.75)
tab_uT=np.ones(len(tab_T))
Regression_MC(tab_z,tab_T,tab_uT)
    
    
    


    
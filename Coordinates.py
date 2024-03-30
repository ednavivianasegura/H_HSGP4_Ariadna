#############################################
#Bibliotecas
from scipy.optimize import fsolve
import warnings
import pandas as pd
import numpy as np
import Angles as A
#from multiprocessing import Pool
import multiprocessing as mp
from sys import exit
#############################################

###################### Comentario Importante #################################
# kepler_solver se utiliza para encontrar la anomalia excentrica "Ea" Esta   #
#funcion debe ir de primeras, para que pueda ser llamada desde otro archivo, #
#si no, genera el error: Can't pickle local object '                         #
##############################################################################

def kepler_solver(X):
    
  def func(Ea):
      e=X[0]
      M=X[1]
      z=Ea-e*np.sin(Ea)-M
      return z
    
  x=X[1]
    
  root = fsolve(func, x)
    
  return root


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CART_ANAL2HYB CONVIERTE COORDENADAS CARTESIANAS (ANALÍTICO + PREDIC. ERROR) en HYBRID.
# Sintaxis:
# hyb = cart_anal2hyb(anal, fore)
#   -anal, fore: matrices que contienen las coordenadas cartesianas.
#       anal: modelo analítico.
#       fore: predicción del error.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].
#   -hyb: matriz que recogerá las coordenadas cartesianas del modelo híbrido. Formato:
#       7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].

def cart_anal2hyb(anal,fore):
    
    # Comprobación tiempos iguales
    
    #ind = int((anal.iloc[:,0:1] != fore.iloc[:,0:1]).sum()) #1° opción
    ind = int((anal["t"] != fore["t"]).sum())
    if (ind!=0):
        warnings.warn("No coinciden los tiempos de ambos conjuntos de efemerides")
        return(0)
    
    # Modelo híbrido. Tiempo igual; resto vars: híbrido = anal. + predic. error
    hyb= pd.concat([anal.iloc[:,0:1], anal.iloc[:,1:7]+fore.iloc[:,1:7]], axis=1)
    hyb.columns=["t","x","y","z","vx","vy","vz"]
    
    return hyb



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CART2DIST CONVIERTE 2 SETS COORD CARTESIANAS -> ERRORES DISTANCIA Y FRENET.
# Sintaxis:
# er_dist = cart2dist(cart_true, cart_test)
#   -cart_true, cart_test: matrices que contienen las coordenadas cartesianas.
#       cart_true: coordenadas precisas (referencia para triedro Frenet).
#       cart_test: coordenadas del modelo evaluado.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].
#   -er_dist: matriz que recogerá los errores en distancia y Frenet. Formato:
#       5 columnas: t[min],dis[km],alo[km],cro[km],rad[km].


def cart2dist(cart_true, cart_test):
  
  # Constantes
  # mu = 398600.4415                           # Constante gravitacional [km^3/s^2]
  
  # Transformaciones
  er_dist = pd.DataFrame(np.zeros((np.shape(cart_true)[0],5)))
  


  # Name variables
  
  cart_true.columns=["t","x","y","z","vx","vy","vz"]
  cart_test.columns=["t","x","y","z","vx","vy","vz"]
  er_dist.columns  =["t","dis","alo","cro","rad"]

  
#  # Coordenadas cartesianas
   
  pos_true = cart_true[["x","y","z"]]
  vel_true = cart_true[["vx","vy","vz"]]
  pos_test = cart_test[["x","y","z"]]


#  # Comprobación tiempos iguales
  
  #ind = int((cart_true.iloc[:,0:1] != cart_test.iloc[:,0:1]).sum()) #1° Opción
  ind = int((cart_true["t"] != cart_test["t"]).sum())
  if (ind!=0):
        warnings.warn("No coinciden los tiempos de ambos conjuntos de efemerides")
        return(0)

  vec_tan = vel_true.div( (((vel_true**2).sum(axis=1)).pow(0.5)),axis='rows') # Tangente: dirección velocidad     
  vec_nor = pd.DataFrame(np.cross(pos_true,vel_true))                         # (pracma package)     
  vec_nor = vec_nor.div((((vec_nor**2).sum(axis=1)).pow(0.5)),axis='rows')    # Normal: direc prod vect pos x vel
  vec_bin=  np.cross(vec_tan, vec_nor)
  
  er_pos= pos_true - pos_test
  
  
  dis= ((er_pos**2).sum(axis=1)).pow(0.5)        # Er dist: módulo vector error posic
  alo= np.multiply(er_pos,vec_tan).sum(axis=1)   # Er along-track: proy (prod esc) dir tan
  cro= np.multiply(er_pos,vec_nor).sum(axis=1)
  rad= np.multiply(er_pos,vec_bin).sum(axis=1) 

  er_dist["t"]= cart_true["t"]
  er_dist["dis"] = dis
  er_dist["alo"] = alo
  er_dist["cro"] = cro
  er_dist["rad"] = rad
  
  return er_dist

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CART2DLN CONVIERTE COORDENADAS CARTESIANAS -> VARIABLES DE DELAUNAY.
# Sintaxis:
# dln = cart2dln(cart)
#   -cart: matriz que contiene las coordenadas cartesianas. Formato:
#   7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].
#   -dln: matriz que recogerá las variables de Delaunay. Formato:
#   7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].


def cart2dln(cart):
  
  # Constantes
  mu = 398600.4415                             # Constante gravitacional [km^3/s^2]
  
  # Transformaciones
  dln =cart .copy()                                   # Tiempo permanece igual
  
  # Name variables
  cart.columns=["t","x","y","z","vx","vy","vz"]
  dln.columns =["t","l","g","h","L","G","H"]

  
  # Coordenadas cartesianas
  pos = cart[["x","y","z"]]                     # Posición
  x   = cart["x"]
  y   = cart["y"]
  z   = cart["z"]
  vel = cart[["vx","vy","vz"]]                  # Velocidad
  
  
  # *****************************************************************
  # TRANSFORMACIONES
  
  # Areal velocity
  am  = np.cross(pos,vel)                       # Angular momentum (pracma package)
  am1 = am[:,0]
  am2 = am[:,1]
  am3 = am[:,2]
   
  AM= np.power(np.power(am,2).sum(axis=1),0.5) # Módulo angular momentum: var. G Delaunay
  # Longitud nodo ascendente
  OM = np.arctan2(am1,-am2)                    # Se utiliza las función de la libreria "Angles"
  OM = A.rem2pi(OM)                                     
  
  # Inclinación
  i= np.arctan2(np.power(np.add(np.power(am1,2),np.power(am2,2)),0.5),am3)
   
  # i = rem2pi(i)
  
  # Argumento latitud
  theta = np.arctan2(z.transpose()*AM,np.add(-x.transpose()*am2,y.transpose()*am1))
  # theta = rem2pi(theta)
  
  # Distancia
  r=np.power((np.power(pos,2)).sum(axis=1),0.5)         # Módulo de pos
  #r = sqrt(rowSums(pos^2))                     
  
  # Semieje mayor
  a=1/(2/r-((np.power(vel,2))/mu).sum(axis=1))
  
  #a=abs(a)  #### Borrar ######
  
  
  #a = 1 / (2/r - rowSums(vel^2)/mu)
  
  # Excentricidad
  eCosE = 1 - r/a                                                     # e*cos(E)
  eSinE = np.multiply(pos,vel).sum(axis=1)/np.power((mu*a),0.5)    # e*sin(E)
  e2    = np.power(eCosE,2) + np.power(eSinE,2)
 
  # Anomalía excéntrica
  
  E=np.arctan2(eSinE,eCosE)
  #E = atan2(eSinE,eCosE)  esta linea esta comentada en R
  
  # Anomalía media
 
  M = E-eSinE 
  M = A.rem2pi(M)                              # Se utiliza las función de la libreria "Angles"
  #M = E-eSinE
  #M = rem2pi(M)
  
  # Anomalía verdadera
  f = np.arctan2(np.multiply(np.power(1-e2,0.5),eSinE), eCosE-e2)
  #f = atan2(sqrt(1-e2)*eSinE, eCosE-e2)
  
  # Argumento perigeo
  w =theta-f
  #w = theta-f
  w = A.rem2pi(w)
  #w = rem2pi(w)
  #%% comentado en el archivo de R.
  # Semiparameter
  # p = a*(1-e^2)
  # p = AM^2 / mu                              # Otra forma
  # *****************************************************************
  #%%
  
  # Variables de Delaunay
  l   = M                                      # Anomalía media:       l=M
  g   = w.transpose()                          # Argumento perigeo:    g=w
  h   = OM.transpose()                         # Longitud nodo asc.:   h=OM
  L   = np.power(mu*a,0.5)
  # L = AM / sqrt(1-e2)                       #Otra forma (esta comentado en R)
  G   = AM.transpose()                         # Módulo angular momentum
  # G = L*sqrt(1-e2)                          # Otra forma (esta comentado en R)
  # G = sqrt(mu*p)                            # Otra forma (esta comentado en R)
  H   = (np.multiply(G,np.cos(i))).transpose()
  
  dln["l"] = l
  dln["g"] = g
  dln["h"] = h
  dln["L"] = L
  dln["G"] = G
  dln["H"] = H
  
  return dln

 #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%
# CART2EQUI CONVIERTE COORDENADAS CARTESIANAS -> ELEMENTOS EQUINOCCIALES.
# Sintaxis:
# equi = cart2equi(cart)
#   -cart: matriz que contiene las coordenadas cartesianas. Formato:
#   7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].
#   -equi: matriz que recogerá los elementos equinocciales. Formato:
#   8 columnas: t[min],a[km],h,k,p,q,lambda[rad],I.


def cart2equi(cart):
    
    equi = orb2equi(cart2orb(cart) )
    return equi

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CART2HILL CONVIERTE COORDENADAS CARTESIANAS -> VARIABLES DE HILL (POLARES-NODALES).
# Sintaxis:
# hill <- cart2hill(cart)
#   -cart: matriz que contiene las coordenadas cartesianas. Formato:
#   7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].
#   -hill: matriz que recogerá las variables de Hill. Formato:
#   7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].


def cart2hill(cart):
  
  # Constantes
  mu = 398600.4415                            # Constante gravitacional [km^3/s^2]
  
  # Transformaciones
  hill = cart.copy()                          # Tiempo permanece igual
  
  # Name variables
  cart.columns  = ["t","x","y","z","vx","vy","vz"]
  hill.columns  = ["t","r","theta","v","R","THETA","N"]
  
  # Coordenadas cartesianas
  pos = cart[["x","y","z"]]       # Posicion
  x = cart["x"]
  y = cart["y"]
  z = cart["z"]
  vel = cart[["vx","vy","vz"]]    # Velocidad
  
  
  # *****************************************************************
  # TRANSFORMACIONES

  # Areal velocity
  am  = np.cross(pos,vel)                          # Angular momentum (pracma package)
  am1 = am[:,0]
  am2 = am[:,1]
  am3 = am[:,2]
  AM  = np.sqrt((np.power(am,2).sum(axis=1)))      # Modulo angular momentum: var. G Delaunay
  
  # Longitud nodo ascendente
  OM =np.arctan2(am1,-am2)
  OM =A.rem2pi(OM)
  
  # Inclinacion
  i =np.arctan2(np.sqrt(np.power(am1,2) + np.power(am2,2)), am3)
    
  # Argumento latitud
  
  theta = np.arctan2(z.transpose()*AM, np.add(-x.transpose()*am2, y.transpose()*am1))
  theta = A.rem2pi(theta)
  
  # Distancia
  r = np.sqrt(np.power(pos,2).sum(axis=1))      # Modulo de pos
  
  # Semieje mayor
  a = 1 / (2/r - (np.power(vel,2)/mu).sum(axis=1))
  
  # Excentricidad
  eCosE = 1 - r/a                              # e*cos(E)
  eSinE = (np.multiply(pos,vel).sum(axis=1))/ np.sqrt(np.multiply(mu,a))        # e*sin(E)
  e2    = np.power(eCosE,2) + np.power(eSinE,2)
  e     = np.sqrt(e2)
  
    
  # AnomalÃ�a verdadera
  f = np.arctan2(np.multiply(np.sqrt(1-e2),eSinE), eCosE-e2)
  
  # G
  G   = AM                               # Modulo angular momentum
  v   = OM                               # Longitud nodo asc.:   v=OM=h
  R   = (np.multiply(mu*e,np.sin(f)))/G  # Velocidad radial:     R
  THETA   = G                            # THETA=G
  N       =np.multiply(G,np.cos(i))      # N=H
  
  hill["r"]      = r
  hill["theta"]  = theta
  hill["v"]      = v
  hill["R"]      = R
  hill["THETA"]  = THETA
  hill["N"]      = N
  
  # Return
  return hill

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CART2ORB CONVIERTE COORDENADAS CARTESIANAS -> ELEMENTOS ORBITALES.
# Sintaxis:
# orb = cart2orb(cart)
#   -cart: matriz que contiene las coordenadas cartesianas. Formato:
#   7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].
#   -orb: matriz que recogerÃ¡ los elementos orbitales. Formato:
#   7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].


def cart2orb(cart):
  
  # Constantes
  mu = 398600.4415                         # Constante gravitacional [km^3/s^2]
  
  # Transformaciones
  orb = cart.copy()                        # Tiempo permanece igual
  
  # Name variables
  cart.columns = ["t","x","y","z","vx","vy","vz"]
  orb.columns  = ["t","a","e","i","OM","w","M"]
  
  # Coordenadas cartesianas
  pos = cart[["x","y","z"]]       # Posicion
  x   = cart["x"]
  y   = cart["y"]
  z   = cart["z"]
  vel = cart[["vx","vy","vz"]]    # Velocidad
  
  
  # *****************************************************************
  # TRANSFORMACIONES
  
  # Areal velocity
  am  = np.cross(pos,vel)                      # Angular momentum (pracma package)
  am1 = am[:,0]
  am2 = am[:,1]
  am3 = am[:,2]
  AM  = np.sqrt(np.power(am,2).sum(axis=1))    # Modulo angular momentum: var. G Delaunay
  
  # Longitud nodo ascendente
  OM = np.arctan2(am1,-am2)
  OM = A.rem2pi(OM)
  
  # Inclinacion
  i = np.arctan2(np.sqrt(np.add(np.power(am1,2),np.power(am2,2))), am3)
  i = A.rem2pi(i)
   
  # Argumento latitud
  theta = np.arctan2(z.transpose()*AM, np.add(-x.transpose()*am2,y.transpose()*am1))
  # theta = rem2pi(theta)
  
  # Distancia
  r = np.sqrt((np.power(pos,2)).sum(axis=1))   # Modulo de pos
  
  # Semieje mayor
  a = 1 / (2/r - (np.power(vel,2).sum(axis=1))/mu)
  
  # Excentricidad
  eCosE = 1 - r/a                              # e*cos(E)
  eSinE = (np.multiply(pos,vel)).sum(axis=1) / np.sqrt(mu*a)        # e*sin(E)
  e2    = np.add(np.power(eCosE,2), np.power(eSinE,2))
  e     = np.sqrt(e2)
  
  # Anomaliaa excentrica
  E = np.arctan2(eSinE,eCosE)
  
  # Anomalia media
  M = E-eSinE
  M = A.rem2pi(M)
  
  # Anomalia verdadera
  f = np.arctan2(np.multiply(np.sqrt(1-e2),eSinE), eCosE-e2)
  
  # Argumento perigeo
  w = theta-f
  w = A.rem2pi(w)
  
  # Elementos orbitales
  orb["a"]   = a                              # Semieje mayor
  orb["e"]   = e                              # Excentricidad
  orb["i"]   = i                              # Inclinacion
  orb["OM"]  = OM                             # Longitud nodo asc.:   OM=h
  orb["w"]   = w                              # Argumento perigeo:    w=g
  orb["M"]   = M                              # Anomalia media:       M=l
  
  return orb

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CART2VEL CONVIERTE 2 SETS COORD CARTESIANAS -> ERRORES VELOCIDAD.
# Sintaxis:
# er_vel = cart2vel(cart_true, cart_test)
#   -cart_true, cart_test: matrices que contienen las coordenadas cartesianas.
#       cart_true: coordenadas precisas (referencia).
#       cart_test: coordenadas del modelo evaluado.
#       Las efemÃ©rides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].
#   -er_vel: vector que recogerÃ¡ los errores en velocidad. Formato:
#       2 columnas: t[min],vel[km/s].


def cart2vel(cart_true, cart_test):
  
  
  # Transformaciones
  er_vel = pd.DataFrame(np.zeros((np.shape(cart_true)[0],2)))
  # Name variables
  cart_true.columns = ["t","x","y","z","vx","vy","vz"]
  cart_test.columns = ["t","x","y","z","vx","vy","vz"]
  er_vel.columns    = ["t","vel"]
  
   
  # Coordenadas cartesianas
  vel_true    = cart_true[["vx","vy","vz"]]
  vel_test    = cart_test[["vx","vy","vz"]]
  
  # Comprobacion tiempos iguales
  t_true = cart_true["t"]
  t_test = cart_test["t"]
  
  ind = int((t_true != t_test).sum())
  if (ind!=0):
        warnings.warn("No coinciden los tiempos de ambos conjuntos de efemerides")
        return(0)
  
 # Vector error velocidad
  er_vel_vec = vel_true - vel_test
  
  # Errores en velocidad
  vel = np.sqrt(np.power(er_vel_vec,2).sum(axis=1))            # Er vel: mÃ³dulo vector error veloc
  
  er_vel["t"]  = t_true                      # Tiempo permanece igual
  er_vel["vel"] = vel
  
  return er_vel

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DLN_ANAL2HYB CONVIERTE VARS. DELAUNAY (ANALITICO + PREDIC. ERROR) -> HYBRID.
# Sintaxis:
# hyb = dln_anal2hyb(anal, fore)
#   -anal, fore: matrices que contienen las variables de Delaunay.
#       anal: modelo analÃ�tico.
#       fore: predicciÃ³n del error.
#       Las efemÃ©rides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].
#   -hyb: matriz que recogerÃ¡ las vars. Delaunay del modelo hÃ�brido. Formato:
#       7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].


def dln_anal2hyb(anal,fore):
  
  # ComprobaciÃ³n tiempos iguales
  ind = int((anal["t"] != fore["t"]).sum())
  if (ind!=0):
        warnings.warn("No coinciden los tiempos de ambos conjuntos de efemerides")
        return(0)
  
  # Modelo hÃ�brido. Tiempo igual; resto vars: hÃ�brido = anal. + predic. error
 
  hyb= pd.concat([anal.iloc[:,0], anal.iloc[:,1:7]+fore.iloc[:,1:7]], axis=1)
    
  hyb.columns = ["t","l","g","h","L","G","H"]
  
  # Variables angulares: normalizaciÃ³n al intervalo [0,2pi)
  hyb["l"] = A.rem2pi(hyb["l"])                 # AnomalÃ�a media:     M=l
  hyb["g"] = A.rem2pi(hyb["g"])                 # Argumento perigeo:  w=g
  hyb["h"] = A.rem2pi(hyb["h"])                 # Longitud nodo asc.: OM=h=v
  
  return hyb

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DLN2CART CONVIERTE VARIABLES DE DELAUNAY -> COORDENADAS CARTESIANAS.
# Sintaxis:
# cart = dln2cart(dln)
#   -dln: matriz que contiene las variables de Delaunay. Formato:
#   7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].
#   -cart: matriz que recogerÃ¡ las coordenadas cartesianas. Formato:
#   7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].

def dln2cart(dln):
  
  cart = orb2cart(dln2orb(dln) )
  
  return cart

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DLN2DIST CONVIERTE 2 SETS VARS. DELAUNAY -> ERRORES DISTANCIA Y FRENET.
# Sintaxis:
# er_dist <- dln2dist(dln_true, dln_test)
#   -dln_true, dln_test: matrices que contienen las variables de Delaunay.
#       dln_true: variables precisas (referencia para triedro Frenet).
#       dln_test: variables del modelo evaluado.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].
#   -er_dist: matriz que recogerá los errores en distancia y Frenet. Formato:
#       5 columnas: t[min],dist[km],along[km],cross[km],radial[km].

def dln2dist(dln_true,dln_test):
  
  er_dist=cart2dist(orb2cart( dln2orb(dln_true)),orb2cart(dln2orb(dln_test)))
 
  return er_dist

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DLN2EQUI CONVIERTE VARIABLES DE DELAUNAY -> ELEMENTOS EQUINOCCIALES.
# Sintaxis:
# equi <- dln2equi(dln)
#   -dln: matriz que contiene las variables de Delaunay. Formato:
#   7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].
#   -equi: matriz que recogerá los elementos equinocciales. Formato:
#   8 columnas: t[min],a[km],h,k,p,q,lambda[rad],I.


def dln2equi(dln):
  
  equi=orb2equi(dln2orb(dln))
  
  return equi

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DLN2HILL CONVIERTE VARIABLES DE DELAUNAY -> VARIABLES DE HILL (POLARES-NODALES).
# Sintaxis:
# hill <- dln2hill(dln)
#   -dln: matriz que contiene las variables de Delaunay. Formato:
#   7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].
#   -hill: matriz que recogerá las variables de Hill. Formato:
#   7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].


def dln2hill(dln):
  
  hill=orb2hill(dln2orb(dln))
  
  return hill
  

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DLN2ORB CONVIERTE VARIABLES DE DELAUNAY -> ELEMENTOS ORBITALES.
# Sintaxis:
# orb = dln2orb(dln)
#   -dln: matriz que contiene las variables de Delaunay. Formato:
#   7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].
#   -orb: matriz que recogerá los elementos orbitales. Formato:
#   7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].

def dln2orb(dln):
 
  # Constantes
  mu = 398600.4415               # Constante gravitacional [km^3/s^2]
  
  # Transformaciones
  orb = dln.copy()                      # Tiempo permanece igual
  
  # Name variables
  dln.columns = ["t","l","g","h","L","G","H"]
  orb.columns = ["t","a","e","i","OM","w","M"]
  
    # Variables de Delaunay
  l = dln["l"]
  g = dln["g"]
  h = dln["h"]
  L = dln["L"]
  G = dln["G"]
  H = dln["H"]
  
  # Elementos orbitales
  a   = (np.power(L,2))/mu                         # Semieje mayor:        a
  e   = np.sqrt(1 - np.power(G,2)/np.power(L,2))   # Excentricidad:        e
  i   = A.rem2pi(np.arccos(H/G))                     # Inclinación:          i
  OM  = A.rem2pi(h)                                  # Longitud nodo asc.:   OM=h
  w   = A.rem2pi(g)                                  # Argumento perigeo:    w=g
  M   = A.rem2pi(l)                                  # Anomalía media:       M=l
  
  orb["a"]   = a
  orb["e"]   = e
  orb["i"]   = i
  orb["OM"]  = OM
  orb["w"]   = w
  orb["M"]   = M
  
  # Return
  return orb

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DLN2VEL CONVIERTE 2 SETS VARS. DELAUNAY -> ERRORES VELOCIDAD.
# Sintaxis:
# er_vel <- dln2vel(dln_true, dln_test)
#   -dln_true, dln_test: matrices que contienen las variables de Delaunay.
#       dln_true: variables precisas (referencia).
#       dln_test: variables del modelo evaluado.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].
#   -er_vel: matriz que recogerá los errores en velocidad. Formato:
#       2 columnas: t[min],vel[km/s].


def dln2vel(dln_true, dln_test):
  
  er_vel=cart2vel(orb2cart(dln2orb(dln_true)),orb2cart(dln2orb(dln_test)))
  
  return er_vel

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EQUI_ANAL2HYB CONVIERTE ELEMENTOS EQUINOCCIALES (ANALÍTICO + PREDIC. ERROR) -> HYBRID.
# Sintaxis:
# hyb <- equi_anal2hyb(anal, fore)
#   -anal, fore: matrices que contienen los elementos equinocciales.
#       anal: modelo analítico.
#       fore: predicción del error.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       -anal: 8 columnas -> t[min],a[km],h,k,p,q,lambda[rad],I.
#       -fore: 8 columnas (I no se utiliza) ó 7 columnas (sin I)->
#           -> t[min],a[km],h,k,p,q,lambda[rad],I.
#   -hyb: matriz que recogerá los elementos equinocciales del modelo híbrido. Formato:
#       8 columnas: t[min],a[km],h,k,p,q,lambda[rad],I.


def equi_anal2hyb(anal,fore):
  
  # Comprobación tiempos iguales
  ind = int((anal["t"] != fore["t"]).sum())
    
  if (ind!=0):
    warnings.warn("No coinciden los tiempos de ambos conjuntos de efemerides")
    return 0
  
  
  # Comprobación formato equi en anal (8 columnas)
  if (np.shape(anal)[1]!=8):
    warnings.warn("Formato incorrecto. La matriz de elementos equinocciales"
                  "del modelo analitico debe tener 8 columnas."
                  "La ultima columna debe contener el factor retrogrado I")
    return 0
  
  
  # Comprobación valores del factor retrógrado en anal
  anal.columns =["t","a","h","k","p","q","lambda","I"]
  
  I = anal["I"]    # Factor retrógrado: I (+1 órb. directa; -1 órb. retróg.)
  ind= ((I!=1) & (I!=-1)).sum()
  
  if (ind!=0):
    warnings.warn("Valor incorrecto del factor retrogrado en el modelo analitico")
    return(0)
  
  
  # Modelo híbrido. Tiempo y factor retrógrado iguales al modelo analítico;
  # resto vars: híbrido = anal. + predic. error
  hyb = pd.concat([anal.iloc[:,0], anal.iloc[:,1:7]+fore.iloc[:,1:7]], axis=1)
  
  hyb.columns=["t","a","h","k","p","q","lambda","I"]
  
  # Variables angulares: normalización al intervalo [0,2pi)
  hyb["lambda"]= A.rem2pi(hyb["lambda"])       # Long media: lambda=M+w+I*OM
  
  # Return
  return hyb

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EQUI2CART CONVIERTE ELEMENTOS EQUINOCCIALES -> COORDENADAS CARTESIANAS.
# Sintaxis:
# cart <- equi2cart(equi)
#   -equi: matriz que contiene los elementos equinocciales. Formato:
#   8 columnas: t[min],a[km],h,k,p,q,lambda[rad],I.
#   -cart: matriz que recogerá las coordenadas cartesianas. Formato:
#   7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].


def equi2cart(equi):
  
  # Comprobación formato equi (8 columnas)
  if (np.shape(equi)[1]!=8):
      warnings.warn("Formato incorrecto. La matriz de elementos equinocciales"
                  "del modelo analitico debe tener 8 columnas."
                  "La ultima columna debe contener el factor retrogrado I")
      return 0
  
  
  # Comprobación valores del factor retrógrado
  equi.columns= ["t","a","h","k","p","q","lambda","I"]
  I = equi["I"]  # Factor retrógrado: I (+1 órb. directa; -1 órb. retróg.)
  
  ind=(abs(I)!=1).sum()
 
  if (ind!=0):
      warnings.warn("Valor incorrecto del factor retrogrado")
      return 0
  
  
  # Conversión
  cart=orb2cart(equi2orb(equi))
  
  return cart

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EQUI2DIST CONVIERTE 2 SETS ELEMENTOS EQUINOCCIALES -> ERRORES DISTANCIA Y FRENET.
# Sintaxis:
# er_dist <- equi2dist(equi_true, equi_test)
#   -equi_true, equi_test: matrices que contienen los elementos equinocciales.
#       equi_true: variables precisas (referencia para triedro Frenet).
#       equi_test: variables del modelo evaluado.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       8 columnas: t[min],a[km],h,k,p,q,lambda[rad],I.
#   -er_dist: matriz que recogerá los errores en distancia y Frenet. Formato:
#       5 columnas: t[min],dist[km],along[km],cross[km],radial[km].


def equi2dist(equi_true, equi_test):
  
  # Comprobación formato equi (8 columnas)
  if(np.shape(equi_true)[1]!=8 or np.shape(equi_test)[1]!=8):
      warnings.warn("Formato incorrecto. Las matrices de elementos equinocciales deben tener 8 columnas. La ultima columna debe contener el factor retrogrado I")
      return 0
  
  
  # Comprobación valores del factor retrógrado: I (+1 órb. directa; -1 órb. retróg.)
  equi_true.columns=["t","a","h","k","p","q","lambda","I"]
  equi_test.columns=["t","a","h","k","p","q","lambda","I"]
  I_true=equi_true["I"]
  I_test=equi_test["I"]
  ind_true=(abs(I_true)!=1).sum()
  ind_test=(abs(I_test)!=1).sum()
  
  if (sum(ind_true)!=0 or sum(ind_test)!=0):
      warnings.warn("Valor incorrecto del factor retrogrado")
      return 0
    
  # Conversión
  er_dist = cart2dist(orb2cart(equi2orb(equi_true)),orb2cart(equi2orb(equi_test)))
  
  return er_dist
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EQUI2DLN CONVIERTE ELEMENTOS EQUINOCCIALES -> VARIABLES DE DELAUNAY.
# Sintaxis:
# dln <- equi2dln(equi)
#   -equi: matriz que contiene los elementos equinocciales. Formato:
#   8 columnas: t[min],a[km],h,k,p,q,lambda[rad],I.
#   -dln: matriz que recogerá las variables de Delaunay. Formato:
#   7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].


def equi2dln(equi):
  
  # Comprobación formato equi (8 columnas)
  if (np.shape(equi)[1]!=8):
      warnings.warn("Formato incorrecto. La matriz de elementos equinocciales debe tener 8 columnas. La ultima columna debe contener el factor retrogrado I")
      return 0
  
  
  # Comprobación valores del factor retrógrado
  equi.columns=["t","a","h","k","p","q","lambda","I"]
  I= equi["I"]  # Factor retrógrado: I (+1 órb. directa; -1 órb. retróg.)
  ind= (abs(I)!=1).sum()
  if (ind!=0):
    warnings.warn("Valor incorrecto del factor retrogrado")
    return 0
  
  
  # Conversión
  dln=orb2dln(equi2orb(equi))
  
  return dln

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EQUI2HILL CONVIERTE ELEMENTOS EQUINOCCIALES -> VARIABLES DE HILL (POLARES-NODALES).
# Sintaxis:
# hill <- equi2hill(equi)
#   -equi: matriz que contiene los elementos equinocciales. Formato:
#   8 columnas: t[min],a[km],h,k,p,q,lambda[rad],I.
#   -hill: matriz que recogerá las variables de Hill. Formato:
#   7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].


def equi2hill(equi):
  
  # Comprobación formato equi (8 columnas)
  if (np.shape(equi)[1]!=8):
      warnings.warn("Formato incorrecto. La matriz de elementos equinocciales debe tener 8 columnas. La ultima columna debe contener el factor retrogrado I")
      return 0
  
  
  # Comprobación valores del factor retrógrado
  equi.columns= ["t","a","h","k","p","q","lambda","I"]
  I = equi["I"]  # Factor retrógrado: I (+1 órb. directa; -1 órb. retróg.)
  ind=(abs(I)!=1).sum()
  if(ind!=0):
      warnings.warn("Valor incorrecto del factor retrogrado")
      return(0)
   
  # Conversión
  hill= orb2hill(equi2orb(equi))
  
  return hill

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EQUI2ORB CONVIERTE ELEMENTOS EQUINOCCIALES -> ELEMENTOS ORBITALES.
# Sintaxis:
# orb <- equi2orb(equi)
#   -equi: matriz que contiene los elementos equinocciales. Formato:
#   8 columnas: t[min],a[km],h,k,p,q,lambda[rad],I.
#   -orb: matriz que recogerá los elementos orbitales. Formato:
#   7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].
  
def equi2orb(equi):
  
  # Comprobación formato equi (8 columnas)
  if (np.shape(equi)[1]!=8):
    warnings.warn("Formato incorrecto. La matriz de elementos equinocciales debe tener 8 columnas. La ultima columna debe contener el factor retrogrado I")
    return(0)
  
  
  # Constantes
  # mu <- 398600.4415                           # Constante gravitacional [km^3/s^2]
  
  # Transformaciones
  orb=equi.iloc[:, 0:7].copy()               # orb sólo 7 cols. Tiempo permanece igual

  # Name variables
  equi.columns  = ["t","a","h","k","p","q","Lambda","I"]
  orb.columns   = ["t","a","e","i","OM","w","M"]
  
  # Elementos equinocciales
  a=equi["a"]    # Semieje mayor:    a
  h=equi["h"]    #                   h
  k=equi["k"]    #                   k
  p=equi["p"]    #                   p
  q=equi["q"]    #                   q
  Lambda=equi["Lambda"]     # Longitud media:   lambda
  I=equi["I"]     # F. retrógrado:    I (+1 órb dir; -1 órb ret)
  
  # Comprobación valores del factor retrógrado
  
  ind=(abs(I)!=1).sum()
  if ind!=0:
      warnings.warn("Valor incorrecto del factor retrogrado")
      return 0
    
  
  # Parámetros auxiliares
  hhkk = np.sqrt(np.add(np.power(h,2),np.power(k,2)))
  ppqq = np.sqrt(np.add(np.power(p,2),np.power(q,2)))
  
  OM                    = np.array(np.arctan2(p,q))
  
  OM[np.where(OM<0)]    = np.add(OM[np.where(OM<0)],2*np.pi)
  
  OM[np.where(ppqq==0)] = 0
  
  zeta      = np.arctan2(h,k)
  ind       = hhkk==0
  zeta[ind] = np.multiply(I[ind], OM[ind])
  
  
  # Elementos orbitales
  # a       <- a                                         # Semieje mayor:        a
  e         = hhkk                                       # Excentricidad:        e
  i         = np.multiply(2,np.arctan(ppqq))             # Inclinación:          i
  i[I==-1]  = np.add(np.pi,-i[I==-1])
  # OM      <- OM                                        # Longitud nodo asc.:   OM=h=v
  w         = A.rem2pi(np.add(zeta,-np.multiply(I,OM)))  # Argumento perigeo:    w=g
  M         = A.rem2pi(np.add(Lambda,-zeta))                      # Anomalía media:       M=l
  
  orb["a"]  = a
  orb["e"]  = e
  orb["i"]  = i
  orb["OM"] = OM
  orb["w"]  = w
  orb["M"]  = M
  
  # Return
  return orb

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EQUI2VEL CONVIERTE 2 SETS ELEMENTOS EQUINOCCIALES -> ERRORES VELOCIDAD.
# Sintaxis:
# er_vel <- equi2vel(equi_true, equi_test)
#   -equi_true, equi_test: matrices que contienen los elementos equinocciales.
#       equi_true: variables precisas (referencia).
#       equi_test: variables del modelo evaluado.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       8 columnas: t[min],a[km],h,k,p,q,lambda[rad],I.
#   -er_vel: matriz que recogerá los errores en velocidad. Formato:
#       2 columnas: t[min],vel[km/s].


def equi2vel(equi_true, equi_test):
  
  # Comprobación formato equi (8 columnas)
  if (np.shape(equi_true)[1]!=8 or np.shape(equi_test)[1]!=8):
      warnings.warn("Formato incorrecto.Las matrices de elementos equinocciales deben tener 8 columnas.La ultima columna debe contener el factor retrogrado I")
      return 0
  
  
  # Comprobación valores del factor retrógrado: I (+1 órb. directa; -1 órb. retróg.)
  
  equi_true.columns =["t","a","h","k","p","q","lambda","I"]
  equi_test.columns= ["t","a","h","k","p","q","lambda","I"]
  I_true   = equi_true["I"]
  I_test   = equi_test["I"]
  ind_true = (abs(I_true)!=1).sum()
  ind_test = (abs(I_test)!=1).sum()
  
  if (ind_true!=0 or ind_test!=0):
    warnings.warn("Valor incorrecto del factor retrogrado")
    return 0
  
  
  # Conversión
  er_vel= cart2vel(orb2cart(equi2orb(equi_true)),orb2cart(equi2orb(equi_test)))
  
  return er_vel

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HILL_ANAL2HYB CONVIERTE VARS. HILL (ANALÍTICO + PREDIC. ERROR) -> HYBRID.
# Sintaxis:
# hyb <- hill_anal2hyb(anal, fore)
#   -anal, fore: matrices que contienen las variables de Hill.
#       anal: modelo analítico.
#       fore: predicción del error.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].
#   -hyb: matriz que recogerá las vars. Hill del modelo híbrido. Formato:
#       7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].


def hill_anal2hyb(anal,fore):
  
  # Comprobación tiempos iguales
  ind = (anal.iloc[:,0] != fore.iloc[:,0]).sum()
  
  if (ind!=0):
    warnings.warn("No coinciden los tiempos de ambos conjuntos de efemerides")
    return 0
  
  
  # Modelo híbrido. Tiempo igual; resto vars: híbrido = anal. + predic. error
  hyb = pd.concat([anal.iloc[:,0], anal.iloc[:,1:7]+fore.iloc[:,1:7]], axis=1)
  hyb.columns= ["t","r","theta","v","R","THETA","N"]
  
  # Variables angulares: normalización al intervalo [0,2pi)
  hyb["theta"]= A.rem2pi(hyb["theta"])         # Arg latitud:  theta=f+w=f+g
  hyb["v"]    = A.rem2pi(hyb["v"])             # Long nod asc: v=OM=h
  
  # Return
  return hyb

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HILL2CART CONVIERTE VARIABLES DE HILL (POLARES-NODALES) -> COORD CARTESIANAS.
# Sintaxis:
# cart <- hill2cart(hill)
#   -hill: matriz que contiene las variables de Hill. Formato:
#   7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].
#   -cart: matriz que recogerá las coordenadas cartesianas. Formato:
#   7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].


def hill2cart(hill):
  
  # Constantes
  # mu <- 398600.4415                   # Constante gravitacional [km^3/s^2]
  
  # Transformaciones
  cart = hill.copy()                          # Tiempo permanece igual
  
  # Name variables
  hill.columns =["t","r","theta","v","R","THETA","N"]
  cart.columns =["t","x","y","z","vx","vy","vz"]
  
  # Variables de Hill
  r     = hill["r"]      # Distancia radial:     r
  theta = hill["theta"]  # Argumento latitud:    theta=f+w=f+g
  v     = hill["v"]      # Longitud nodo asc.:   v=OM=h
  R     = hill["R"]      # Velocidad radial:     R
  THETA = hill["THETA"]  #                       THETA=G
  N     = hill["N"]      #                       N=H
  
  # Funciones trigonométricas
  # i   <- acos(N/THETA)               # Inclinación:  H=G*cos(i), N=THETA*cos(i)
  ci    = N/THETA                      # ci=cos(i)
  si    = (1-ci**2).pow(0.5)           # si=sin(i)
  # si  <- sin(acos(ci))               # Otra forma
  ct    = np.cos(theta)                # ct=cos(theta)
  st    = np.sin(theta)                # st=sin(theta)
  cv    = np.cos(v)                    # cv=cos(v)
  sv    = np.sin(v)                    # sv=sin(v)
  
  # Coordenadas cartesianas
  x   = r * (ct*cv - ci*st*sv)                                 # x
  y   = r * (ct*sv + ci*st*cv)                                 # y
  z   = r*si*st                                                # z
  vx  = (R*ct - THETA*st/r)*cv - (R*st + THETA*ct/r)*sv*ci     # vx
  vy  = (R*ct - THETA*st/r)*sv + (R*st + THETA*ct/r)*cv*ci     # vy
  vz  = (R*st + THETA*ct/r)*si                                 # vz
  
  cart["x"]  = x
  cart["y"]  = y
  cart["z"]  = z
  cart["vx"] = vx
  cart["vy"] = vy
  cart["vz"] = vz
  
  # Return
  return cart

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HILL2DIST CONV. 2 SETS VAR HILL (POLARES-NODALES) -> ERRORES DIST Y FRENET.
# Sintaxis:
# er_dist <- hill2dist(hill_true, hill_test)
#   -hill_true, hill_test: matrices que contienen las variables de Hill.
#       hill_true: variables precisas (referencia para triedro Frenet).
#       hill_test: variables del modelo evaluado.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].
#   -er_dist: matriz que recogerá los errores en distancia y Frenet. Formato:
#       5 columnas: t[min],dist[km],along[km],cross[km],radial[km].


def hill2dist(hill_true, hill_test):
  
  er_dist = cart2dist(hill2cart(hill_true),hill2cart(hill_test))
  
  return er_dist




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HILL2DLN CONVIERTE VARIABLES DE HILL (POLARES-NODALES) -> VARIABLES DE DELAUNAY.
# Sintaxis:
# dln <- hill2dln(hill)
#   -hill: matriz que contiene las variables de Hill. Formato:
#   7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].
#   -dln: matriz que recogerá las variables de Delaunay. Formato:
#   7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].


def hill2dln(hill):
  
  dln = cart2dln(hill2cart(hill))
  
  return dln




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HILL2EQUI CONVIERTE VARIABLES DE HILL (POLARES-NODALES) -> ELEMENTOS EQUINOCCIALES.
# Sintaxis:
# equi <- hill2equi(hill)
#   -hill: matriz que contiene las variables de Hill. Formato:
#   7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].
#   -equi: matriz que recogerá los elementos equinocciales. Formato:
#   8 columnas: t[min],a[km],h,k,p,q,lambda[rad],I.


def hill2equi(hill):
  
  equi=orb2equi(hill2orb(hill))
  
  return equi

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HILL2ORB CONVIERTE VARIABLES DE HILL (POLARES-NODALES) -> ELEMENTOS ORBITALES.
# Sintaxis:
# orb <- hill2orb(hill)
#   -hill: matriz que contiene las variables de Hill. Formato:
#   7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].
#   -orb: matriz que recogerá los elementos orbitales. Formato:
#   7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].


def hill2orb(hill):
  
  orb = cart2orb(hill2cart(hill))
  
  return orb
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HILL2VEL CONV. 2 SETS VAR HILL (POLARES-NODALES) -> ERRORES VELOCIDAD.
# Sintaxis:
# er_vel <- hill2vel(hill_true, hill_test)
#   -hill_true, hill_test: matrices que contienen las variables de Hill.
#       hill_true: variables precisas (referencia).
#       hill_test: variables del modelo evaluado.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].
#   -er_vel: matriz que recogerá los errores en velocidad. Formato:
#       2 columnas: t[min],vel[km/s].


def hill2vel(hill_true, hill_test):
  
  er_vel = cart2vel(hill2cart(hill_true),hill2cart(hill_test))
  
  return er_vel

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ORB_ANAL2HYB CONVIERTE ELEMENTOS ORBITALES (ANALÍTICO + PREDIC. ERROR) -> HYBRID.
# Sintaxis:
# hyb <- orb_anal2hyb(anal, fore)
#   -anal, fore: matrices que contienen los elementos orbitales.
#       anal: modelo analítico.
#       fore: predicción del error.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].
#   -hyb: matriz que recogerá los elementos orbitales del modelo híbrido. Formato:
#       7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].


def orb_anal2hyb(anal, fore):
    
    
      # Comprobación tiempos iguales
  ind = (anal.iloc[:,0] != fore.iloc[:,0]).sum()
  
  if (ind!=0):
    warnings.warn("No coinciden los tiempos de ambos conjuntos de efemerides")
    return 0
 
  # Modelo híbrido. Tiempo igual; resto vars: híbrido = anal. + predic. error
  hyb = pd.concat([anal.iloc[:,0], anal.iloc[:,1:7]+fore.iloc[:,1:7]], axis=1)
  
  hyb.columns = ["t","a","e","i","OM","w","M"]
  
  # Variables angulares: normalización al intervalo [0,2pi)
  hyb["i"]  = A.rem2pi(hyb["i"])         # Inclinación:        i
  hyb["OM"] = A.rem2pi(hyb["OM"])        # Longitud nod asc.:  OM=h=v
  hyb["w"]  = A.rem2pi(hyb["w"])         # Argumento perigeo:  w=g
  hyb["M"]  = A.rem2pi(hyb["M"])         # Anomalía media:     M=l
  
  # Return
  return hyb

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ORB_M2R CAMBIA ORDEN ELEMENTOS ORBITALES: MATLAB -> R.
# Sintaxis:
# orb_R <- orb_M2R(orb_M)
#   -orb_M: matriz que contiene elementos orbitales (orden Matlab). Formato:
#   7 columnas: t[min],a[km],e,i[rad],w[rad],OM[rad],M[rad].
#   -orb_R: matriz que recogerá elementos orbitales (orden R). Formato:
#   7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].


def orb_M2R(orb_M):
  
  orb_R= orb_M.copy()
  orb_M.columns = ["t","a","e","i","w","OM","M"]
  orb_R.columns = ["t","a","e","i","OM","w","M"]
  orb_R["OM"]   = orb_M["OM"]
  orb_R["w"]    = orb_M["w"]
  return orb_R

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ORB_R2M CAMBIA ORDEN ELEMENTOS ORBITALES: R -> MATLAB.
# Sintaxis:
# orb_M <- orb_R2M(orb_R)
#   -orb_R: matriz que contiene elementos orbitales (orden R). Formato:
#   7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].
#   -orb_M: matriz que recogerá elementos orbitales (orden Matlab). Formato:
#   7 columnas: t[min],a[km],e,i[rad],w[rad],OM[rad],M[rad].


def orb_R2M(orb_R):
  
  orb_M= orb_R.copy()
  orb_R.colnames =["t","a","e","i","OM","w","M"]
  orb_M.colnames =["t","a","e","i","w","OM","M"]
  orb_M["w"]     =orb_R["w"]
  orb_M["OM"]    = orb_R["OM"]
  
  return orb_M

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ORB2CART CONVIERTE ELEMENTOS ORBITALES -> COORDENADAS CARTESIANAS.
# Sintaxis:
# cart = orb2cart(orb)
#   -orb: matriz que contiene los elementos orbitales. Formato:
#   7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].
#   -cart: matriz que recogerÃ¡ las coordenadas cartesianas. Formato:
#   7 columnas: t[min],x[km],y[km],z[km],vx[km/s],vy[km/s],vz[km/s].


def orb2cart(orb):
  
  # Constantes
  mu = 398600.4415                             # Constante gravitacional [km^3/s^2]
  
  # Transformaciones
  cart = orb.copy()                                   # Tiempo permanece igual
  
  # Name variables
  orb.columns  = ["t","a","e","i","OM","w","M"]
  cart.columns = ["t","x","y","z","vx","vy","vz"]
  
  # Elementos orbitales
  a   = orb["a"]
  e   = orb["e"]
  i   = orb["i"]
  OM  = orb["OM"]                 # Longitud nodo asc.:   OM=h=v
  w   = orb["w"]                  # Argumento perigeo:    w=g
  M   = orb["M"]                  # Anomalia media:       M=l
  
  # Anomalia excentrica: Ea (Resol eq Kepler)
  #Ea = pd.DataFrame(np.zeros((np.shape(M)[0],1)))
  
  eM= pd.concat([e,M], axis=1)                       # Parejas (e,M) -> Ea (eq Kepler)

  pool=mp.Pool(mp.cpu_count())

  # Define the dataset

  dataset = eM.values.tolist()
  
  result=pool.starmap(kepler_solver,[[row] for row in dataset])
  
  pool.close()
   
  Ea=np.concatenate( result, axis=0 )
 
  # Transformaciones
  cosE= np.cos(Ea)
  sinE= np.sin(Ea)
  # Perifocal coordinates
  fac   = np.sqrt(1-np.power(e,2))
  R     = a*(1 - e*cosE.transpose())                 # Distance
  V     = np.sqrt(np.multiply(mu,a))/R               # Velocity
  xp    = a*(cosE.transpose()-e)
  yp    = (a*fac)*sinE.transpose()
  # zp  =  matrix(0,nrow(xp),1)                # No utilizada
  xpd   = (-V)*sinE.transpose()
  ypd   = (V*fac.transpose())*cosE.transpose()
  # zpd =  matrix(0,nrow(xpd),1)               # No utilizada
  
  # Now compute remaining pos/vel coordinates
  ci    = np.cos(i).transpose()                              # Cos, sin of inclination
  si    = np.sin(i).transpose()
  clan  = np.cos(OM).transpose()                             # Cos, sin of longitude of ascending node
  slan  = np.sin(OM).transpose()
  cw    = np.cos(w).transpose()                              # Cos, sin of arg of perigee
  sw    = np.sin(w).transpose()
  
  # Compute matrix to convert from perifocal to geocentric-equatorial coordinates
  r11   =  np.subtract(np.multiply(cw,clan), np.multiply(np.multiply(sw,slan),ci))
  r21   =  np.add(np.multiply(cw,slan), np.multiply(np.multiply(sw,clan),ci))
  r31   =  np.multiply(sw,si)
  r12   =  np.add(-np.multiply(sw,clan), - np.multiply(np.multiply(cw,slan),ci))
  r22   =  np.add(-np.multiply(sw,slan), np.multiply(np.multiply(cw,clan),ci))
  r32   =  np.multiply(cw,si)
  
  # Now apply the matrix to compute the output position and velocity
  # Coordenadas cartesianas

  x     =  r11*xp+r12*yp     # x
  y     =  r21*xp+r22*yp     # y
  z     =  r31*xp+r32*yp     # z
  vx    =  r11*xpd+r12*ypd   # vx
  vy    =  r21*xpd+r22*ypd   # vy
  vz    =  r31*xpd+r32*ypd   # vz
  
  cart["x"]  = x.transpose()
  cart["y"]  = y.transpose()
  cart["z"]  = z.transpose()
  cart["vx"] = vx.transpose()
  cart["vy"] = vy.transpose()
  cart["vz"] = vz.transpose()
  
  return cart

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ORB2DIST CONVIERTE 2 SETS ELEMS. ORBITALES -> ERRORES DISTANCIA Y FRENET.
# Sintaxis:
# er_dist <- orb2dist(orb_true, orb_test)
#   -orb_true, orb_test: matrices que contienen los elementos orbitales.
#       orb_true: elementos precisos (referencia para triedro Frenet).
#       orb_test: elementos del modelo evaluado.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].
#   -er_dist: matriz que recogerá los errores en distancia y Frenet. Formato:
#       5 columnas: t[min],dist[km],along[km],cross[km],radial[km].


def orb2dist(orb_true, orb_test):
  
  er_dist=cart2dist(orb2cart(orb_true),orb2cart(orb_test))
  
  return er_dist
  
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ORB2DLN CONVIERTE ELEMENTOS ORBITALES -> VARIABLES DE DELAUNAY.
# Sintaxis:
# dln <- orb2dln(orb)
#   -orb: matriz que contiene los elementos orbitales. Formato:
#   7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].
#   -dln: matriz que recogerá las variables de Delaunay. Formato:
#   7 columnas: t[min],l[rad],g[rad],h[rad],L[km^2/s],G[km^2/s],H[km^2/s].


def orb2dln(orb):
  
  # Constantes
  mu = 398600.4415               # Constante gravitacional [km^3/s^2]
  
  # Transformaciones
  dln = orb.copy()                      # Tiempo permanece igual
  
  # Name variables
  orb.columns = ["t","a","e","i","OM","w","M"]
  dln.columns = ["t","l","g","h","L","G","H"]
  
  # Elementos orbitales
  a   =orb["a"]
  e   =orb["e"]
  i   =orb["i"]
  OM  =orb["OM"]
  w   =orb["w"]
  M   =orb["M"]
  
  # Variables de Delaunay
  l = A.rem2pi(M)                  # Anomalía media:       l=M
  g = A.rem2pi(w)                  # Argumento perigeo:    g=w
  h = A.rem2pi(OM)                 # Longitud nodo asc.:   h=OM
  L = (mu*a).pow(0.5)
  G = L*((1-e**2).pow(0.5))
  H = G*np.cos(i)
  
  dln["l"]= l
  dln["g"]= g
  dln["h"]=h
  dln["L"]=L
  dln["G"]=G
  dln["H"]=H
  
  # Return
  return dln

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ORB2EQUI CONVIERTE ELEMENTOS ORBITALES -> ELEMENTOS EQUINOCCIALES.
# Sintaxis:
# equi <- orb2equi(orb)
#   -orb: matriz que contiene los elementos orbitales. Formato:
#   7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].
#   -equi: matriz que recogerá los elementos equinocciales. Formato:
#   8 columnas: t[min],a[km],h,k,p,q,lambda[rad],I.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def orb2equi(orb):
    
    # Constantes
    # mu =398600
    # Transformaciones
    
    equi = pd.DataFrame(np.zeros(((np.shape(orb)[0]),8)))
    
    # Name variables
    
    orb.columns  =["t","a","e","i","OM","w","M"]
    equi.columns =["t","a","h","k","p","q","lambda","I"]
    
    # Elementos orbitales
    
    t= orb["t"]
    a= orb["a"]
    e= orb["e"]
    i= orb["i"]
    OM= orb["OM"]   # Longitud nodo asc. : OM=h=v
    w= orb["w"]     # Argumento perigeo:   w=g
    M= orb["M"]     # Anomalia medi
    
###############
    # FACTOR RETROGRADO: I
    
    I= np.array([0]*len(i))
    
    # Normalizar i al intervalo [0,2pi)

    i = A.rem2pi(i)
    
    # Órbita directa (i en cuadrantes 1 ó 4): I = +1
    
    ind = ((i>=0) & (i<=np.pi/2)) | ((i>=3*np.pi/2) & (i<=2*np.pi))
    
    I[ind]= 1
    
     # Órbita retrógrada (i en cuadrantes 2 ó 3): I = -1
    
    ind = (i>np.pi/2) & (i<3*np.pi/2)
    
    I[ind]= -1
        
    # Comprobación valores del factor retrógrado
    
    ind= (I!=1) & (I!=-1)
    
    if (sum(ind)!=0):
        warnings.warn("Error en el factor retrogrado. Valor incorrecto de la inclinacion")
        return(0)
  
# %%
  
   # Elementos equinocciales
   # t     = t                           # Tiempo permanece igual
   # a     = a                           # Semieje mayor:        a
    h = np.multiply(e,np.sin(np.add(w, np.multiply(I,OM))))       #                       h
    k = np.multiply(e,np.cos(np.add(w, np.multiply(I,OM))))       #                       k
    p = np.multiply(np.power(np.tan(i/2),I),np.sin(OM))           #                       p
    q = np.multiply(np.power(np.tan(i/2),I),np.cos(OM))           #                       q
    Lambda = A.rem2pi(M+w+np.multiply(I,OM))                      # Longitud media:       lambda
    # I     = I                                                   # Factor retrógrado:    I
  
    equi["t"] = t
    equi["a"] = a
    equi["h"] = h
    equi["k"] = k
    equi["p"] = p
    equi["q"] = q
    equi["lambda"] = Lambda
    equi["I"]      = I
  
    return(equi)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ORB2HILL CONVIERTE ELEMENTOS ORBITALES -> VARIABLES DE HILL (POLARES-NODALES).
# Sintaxis:
# hill <- orb2hill(orb)
#   -orb: matriz que contiene los elementos orbitales. Formato:
#   7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].
#   -hill: matriz que recogerá las variables de Hill. Formato:
#   7 columnas: t[min],r[km],theta[rad],v[rad],R[km/s],THETA[km^2/s],N[km^2/s].


def orb2hill(orb):
  
  # Constantes
  mu = 398600.4415                            # Constante gravitacional [km^3/s^2]
  
  # Transformaciones
  hill = orb.copy()                           # Tiempo permanece igual
  
  # Name variables
  orb.columns   = ["t","a","e","i","OM","w","M"]
  hill.columns  = ["t","r","theta","v","R","THETA","N"]
  
  # Elementos orbitales
  a   = orb["a"]
  e   = orb["e"]
  i   = orb["i"]
  OM  = orb["OM"]                 # Longitud nodo asc.:   OM=h=v
  w   = orb["w"]                  # Argumento perigeo:    w=g
  M   = orb["M"]                  # Anomalía media:       M=l
  
  
  # *****************************************************************
  # Anomalía excéntrica: Ea (Resol eq Kepler)
  #Ea = pd.DataFrame(np.zeros((np.shape(M)[0],1)))
  
  eM= pd.concat([e,M], axis=1)                       # Parejas (e,M) -> Ea (eq Kepler)
  
  pool=mp.Pool(mp.cpu_count())

  # Define the dataset

  dataset = eM.values.tolist()
  
  result=pool.starmap(kepler_solver,[[row] for row in dataset])
  
  pool.close()
   
  Ea=np.concatenate( result, axis=0 )  
 
  # Transformaciones
  f = 2 * np.arctan(np.sqrt((1+e)/(1-e))*(np.tan(Ea/2).transpose()))    # Anomalía verdadera:   f
  p = a * (1-e**2)                                        # Semiparameter:        p
  G = np.sqrt(mu*p)                                        #                       G
  
  # Variables de Hill
  r     = p/(1 + e*np.cos(f))                             # Dist. radial:         r
  theta = A.rem2pi(np.add(f,w))                                   # Argumento latitud:    theta=f+w=f+g
  v     = A.rem2pi(OM)                                    # Longitud nodo asc.:   v=OM=h
  R     = mu*e*np.sin(f) / G                              # Velocidad radial:     R
  THETA = G                                               #                       THETA=G
  N     = G*np.cos(i)                                     #                       N=H
  
  hill["r"]      = r.transpose()  
  hill["theta"] = theta
  hill["v"]     = v
  hill["R"]     = R.transpose()  
  hill["THETA"] = THETA
  hill["N"]     = N
  
  # Return
  return hill

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ORB2VEL CONVIERTE 2 SETS ELEMS. ORBITALES -> ERRORES VELOCIDAD.
# Sintaxis:
# er_vel <- orb2vel(orb_true, orb_test)
#   -orb_true, orb_test: matrices que contienen los elementos orbitales.
#       orb_true: elementos precisos (referencia).
#       orb_test: elementos del modelo evaluado.
#       Las efemérides deben corresponder a los mismos instantes. Formato:
#       7 columnas: t[min],a[km],e,i[rad],OM[rad],w[rad],M[rad].
#   -er_vel: matriz que recogerá los errores en velocidad. Formato:
#       2 columnas: t[min],vel[km/s].


def orb2vel(orb_true, orb_test):
  
  er_vel=cart2vel(orb2cart(orb_true),orb2cart(orb_test))
  return er_vel

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def hyb_distFun(argument,obs_testC,appr_testC,fore_testC):
    
    if argument not in ["dln", "orb", "equi", "hill", "cart"]:
         
        exit("Unknown variable set ("+ str(argument) + "). Possibilities: dln, orb, equi, hill, cart")

    else:
        
        if   argument== 'hill':
            return hill2dist(obs_testC,hill_anal2hyb(appr_testC, fore_testC))
        
        elif argument== 'orb' :
            return orb2dist (obs_testC,orb_anal2hyb (appr_testC, fore_testC))

        elif argument== 'equi' :
            return equi2dist(obs_testC,equi_anal2hyb(appr_testC, fore_testC))

        elif argument== 'dln'  :
            return dln2dist (obs_testC,dln_anal2hyb (appr_testC, fore_testC))
        
        else:
            return cart2dist(obs_testC,cart_anal2hyb(appr_testC, fore_testC))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    
def appr_distFun(argument,obs_testC,appr_testC):
    
    if argument not in ["dln", "orb", "equi", "hill", "cart"]:
         
        exit("Unknown variable set ("+ str(argument) + "). Possibilities: dln, orb, equi, hill, cart")
    
    else:
        
        if   argument== 'hill':
            return hill2dist(obs_testC, appr_testC)
        
        elif argument== 'orb' :
            return orb2dist( obs_testC, appr_testC)

        elif argument== 'equi' :
            return equi2dist(obs_testC, appr_testC)

        elif argument== 'dln'  :
            return dln2dist( obs_testC, appr_testC)
        
        else:
            return cart2dist(obs_testC, appr_testC)
    

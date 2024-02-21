# -*- coding: utf-8 -*-
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# rem RESTO CON SIGNO DE LA DIVISIÓN.
# Sintaxis: out = rem(x, y)
#   -x: dividendo.
#   -y: divisor.
#   -out: resto con signo de la divisi�n x/y.
#
# Equivalente a la función rem de Matlab.

#%%
import numpy as np

def rem(x, y):
    X=np.array(x)
    if (y!=0):
        n=np.trunc(X/y)
        out= X - n*y
  
    else:
        out=float('Inf')

    return out

#%%
 
# %%
# rem2pi_sym NORMALIZA MAGNITUDES ANGULARES AL INTERVALO (-pi,pi].
# Sintaxis: out = rem2pi_sym(x)
#   -x: datos de partida.
#   -out: datos normalizados.
#
# Normaliza al intervalo SYMmetrical (-pi,pi], que es más conveniente que [0,2pi)
# para magnitudes próximas a cero, como sucede cuando se manejan errores, pues
# conserva su continuidad en el entorno de cero.
# 
# La función puede ser aplicada a cualquier matriz o vector.
# No aplicar a matrices que incluyan magnitudes no angulares,
# pues modifica cualquier valor no comprendido en (-pi,pi].
 
# %%

def rem2pi_sym(x):
    dosPi=2*np.pi
    out                          = rem(x, dosPi)                              # (-2pi,2pi)
    out[np.where(out>np.pi)]     = out[np.where(out>np.pi)] - dosPi           # (-2pi,pi]
    out[np.where(out <= -np.pi)] = out[np.where(out <= -np.pi)] + dosPi       # (-pi,pi]
    return out

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# rem2pi NORMALIZA MAGNITUDES ANGULARES AL INTERVALO [0,2pi).
# Sintaxis: out = rem2pi(x)
#   -x: datos de partida.
#   -out: datos normalizados.
#
# La funci�n puede ser aplicada a cualquier matriz o vector.
# No aplicar a matrices que incluyan magnitudes no angulares,
# pues modifica cualquier valor no comprendido en [0,2pi).


def rem2pi(x):
    dosPi=2*np.pi
    out         =rem(x, dosPi)                            # (-2pi,2pi)
    out[np.where(out<0)]  = out[np.where(out<0)] + dosPi  # [0,2pi)
    return out

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# remove2pi TRANSFORMA VALORES PR�XIMOS A +/-(2*pi) A VALORES PR�XIMOS A 0.
# Sintaxis: out = remove2pi(x, holgura)
#   -x: datos de partida.
#   -holgura: todos los valores comprendidos entre +/-(2*pi)-holgura y
#     +/-(2*pi)+holgura ser�n transformados rest�ndoles/sum�ndoles 2*pi.
#   -out: datos transformados.
#
# La funci�n puede ser aplicada a cualquier matriz o vector.
# Vigilar el resultado si se aplica a matrices que incluyen magnitudes
# no angulares, pues modifica cualquier valor pr�ximo a 6.28 � a -6.28.


def remove2pi(x, holgura):
    dosPi     = 2*np.pi
    out       = np.array(x)
    ind       = np.where((out<= dosPi + holgura) & (out>= dosPi - holgura))
    out[ind]  = out[ind] - dosPi
    ind       = np.where((out<= -dosPi + holgura) & (out>= -dosPi - holgura))
    out[ind]  = out[ind] + dosPi
    return out



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# rem360_sym NORMALIZA MAGNITUDES ANGULARES AL INTERVALO (-180,180].
# Sintaxis: out = rem360_sym(x)
#   -x: datos de partida.
#   -out: datos normalizados.
#
# Normaliza al intervalo SYMmetrical (-180,180], que es m�s conveniente que [0,360)
# para magnitudes pr�ximas a cero, como sucede cuando se manejan errores, pues
# conserva su continuidad en el entorno de cero.
# 
# La funci�n puede ser aplicada a cualquier matriz o vector.
# No aplicar a matrices que incluyan magnitudes no angulares,
# pues modifica cualquier valor no comprendido en (-180,180].


def rem360_sym(x):
    out                         = rem(x, 360)                          # (-360,360)
    out[np.where(out >   180)]  = out[np.where(out > 180)]   - 360     # (-360,180]
    out[np.where(out <= -180)]  = out[np.where(out <= -180)] + 360     # (-180,180]
    return out

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# rem360 NORMALIZA MAGNITUDES ANGULARES AL INTERVALO [0,360).
# Sintaxis: out = rem360(x)
#   -x: datos de partida.
#   -out: datos normalizados.
#
# La funci�n puede ser aplicada a cualquier matriz o vector.
# No aplicar a matrices que incluyan magnitudes no angulares,
# pues modifica cualquier valor no comprendido en [0,360).


def rem360(x):
    out                   = rem(x, 360)                       # (-360,360)
    out[np.where(out<0)]  = out[np.where(out<0)] + 360        # [0,360)
    return out

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# remove360 TRANSFORMA VALORES PR�XIMOS A +/-(360) A VALORES PR�XIMOS A 0.
# Sintaxis: out = remove360(x, holgura)
#   -x: datos de partida.
#   -holgura: todos los valores comprendidos entre +/-(360)-holgura y
#     +/-(360)+holgura ser�n transformados rest�ndoles/sum�ndoles 360.
#   -out: datos transformados.
#
# La funci�n puede ser aplicada a cualquier matriz o vector.
# Vigilar el resultado si se aplica a matrices que incluyen magnitudes
# no angulares, pues modifica cualquier valor pr�ximo a 360 � a -360.


def remove360(x, holgura):
  out       = np.array(x)
  ind       = np.where((out <= 360 + holgura)  & (out >= 360 - holgura))
  out[ind]  = out[ind] - 360
  ind       = np.where((out <= -360 + holgura) & (out >= -360 - holgura))
  out[ind]  = out[ind] + 360
  return out

# 


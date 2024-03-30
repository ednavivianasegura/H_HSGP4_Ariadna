# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:05:25 2021

# Versión completa I (FUNCIONES)

# ARIADNA HYBRID PROPAGATION
# MODELING & FORECASTING

@author: edsegualv && hacarrh
"""
import itertools, sys, os, time, gc, shutil
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
###########################################################
######## Scripts saved in the CodigosPython folder ########
###########################################################
import Coordinates as coor
# import CodigosPython.Period as pe
import Angles as an
import ml_misc as ml_misc
###########################################################
######## Pronóstico c++ Libraries ########
###########################################################
import subprocess
import platform
Plataforma=platform.platform()
###########################################################
if not Plataforma.find("indows") == 1: 
    import matplotlib
    matplotlib.use('Agg') #Only activate if running in backend (linux), 
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
###########################################################
######## Deep Learning Libraries ########
###########################################################
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import KFold


def RMSE(Reales,Fore):
    diff=np.subtract(Reales,Fore)
    square=np.square(diff)
    MSE=square.mean()
    RMSE=np.sqrt(MSE)
    print("Root Mean Square Error:", RMSE)
    return RMSE



###########################################################
######## Pronóstico c++ Functions                   #######
###########################################################
#Function in charge of creating the file named "LeerPesosdetxt.h" which is in charge of reading the weights of the model to be forecasted.
def LeerPesosdetxt(path,path_LeerPesosdetxt):
    file=path+path_LeerPesosdetxt+"LeerPesosdetxt.h"
    try:
        os.remove(file)
        print(str(file)+" Be created again")
    except:
        print(str(file)+" Initially created")
    #Comienza a escribir en el archivo 
    with open(file, 'a') as f:
           print(
               "#include<Eigen/Dense>\n"
               "#include<vector>\n"
               "using namespace std;\n"
               "Eigen::MatrixXd openData(std::string fileToOpen)\n"
               "{\n"
 
    "// the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)\n"
    "// https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix\n"
     
    '// the input is the file: "fileToOpen.csv":\n'
    "// a,b,c\n"
    "// d,e,f\n"
    "// This function converts input file data into the Eigen matrix format\n"
    "// the matrix entries are stored in this variable row-wise. For example if we have the matrix:\n"
    "// M=[a b c \n"
    "//    d e f]\n"
    '// the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector\n'
    "// later on, this vector is mapped into the Eigen matrix format\n"
    "std::vector<double> matrixEntries;\n"
     "// in this object we store the data from the matrix\n"
    "std::ifstream matrixDataFile(fileToOpen);\n"
     "// this variable is used to store the row of the matrix that contains commas\n"
    "std::string matrixRowString;\n"
    "// this variable is used to store the matrix entry;\n"
    "std::string matrixEntry;\n"
     "// this variable is used to track the number of rows\n"
    "int matrixRowNumber = 0;\n" 
    "// here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString\n"
    "while (getline(matrixDataFile, matrixRowString))\n"
    "{\n"
        "//convert matrixRowString that is a string to a stream variable.\n"
        "stringstream matrixRowStringStream(matrixRowString);\n"
        "// here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry\n"
        "while (getline(matrixRowStringStream, matrixEntry, ','))\n"
        "{\n"
	"//here we convert the string to double and fill in the row vector storing all the matrix entries\n"
            "matrixEntries.push_back(stod(matrixEntry));\n"
        "}\n"

	"//update the column numbers\n"
        "matrixRowNumber++;\n"
    "}\n"
 
    "/* here we convet the vector variable into the matrix and return the resulting object,note that matrixEntries.data()\n"
       "is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;*/\n"
    
    "return Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,\n"
				    "Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);\n"
 
"}\n"              
             , file=f)

#Function in charge of extracting the configuration of the model to be forecasted
def ExtractModInfo(Mod, ResumenEjecucion):
    Plataforma=platform.platform()
    CONFIGURACION=Mod.get_config()
    capas=CONFIGURACION['layers']
    NCapas=str(capas).count("Dense")
    print("Plataforma: \n")
    print(Plataforma)
    try:
        #Saves the names of all activation functions:
        Act=[capas[i]['config']['activation'] for i in range(1,NCapas+1)] 
    except:   
        Act=[capas[i]['config']['activation'] for i in range(1,NCapas)]         
     
    #Save the names for the weights and biases matrixes:  
    K=[l+str(i+1) for i in range(NCapas) for l in ["W","B"]]          
    
    with open(ResumenEjecucion) as f:
        if str("Activation functions: "+ str(Act)) in f.read():
            escribir=False
        else:
            escribir=True
    if escribir:
        with open(ResumenEjecucion, 'a') as f:
                print("###########################################################\n"+
                      "NCapas: "+ str(NCapas)+" \n"+
                      "Activation functions: "+ str(Act)+" \n"+
                      "Names for theweights and biases matrixes : "+ str(K)+" \n"+
                      "###########################################################\n"
                  , file=f)   
    
    return NCapas,Act,K    

#Function that creates RedNeuronal.cpp file:
def creaRedNeuronalcMasMas(FileName,K,Act,NCapas,Functions):
    try:
        os.remove(FileName)
        print(str(FileName)+" Be created again")
    except:
        print(str(FileName)+" Initially created")
    #Comienza a escribir en el archivo 
    with open(FileName, 'a') as f:
           print("#include <iostream>\n"
                 "#include<iomanip>\n"
                 "#include<Eigen/Dense>\n"
                 "#include<fstream>\n"
                 '#include "LeerPesosdetxt.h"\n'
                 "//***********************************************************\n"
                 "\n\n"
                 "// matriz de datos de entrada:\n"
                 "Eigen::MatrixXd X;\n"             
             , file=f)
       
    for i in K:
        with open(FileName, 'a') as f:
            if i[0]=="W":
                print("//Matriz capa "+str(i[1])+":\n"+
                      "Eigen::MatrixXd " + i + ";"
                      ,file=f)
                print("//Matriz multiplicación número "+str(i[1])+":\n"+
                      "Eigen::MatrixXd X_" + i + ";"
                      ,file=f)
            else:
                print("//Sesgo capa "+str(i[1])+":\n"+
                      "Eigen::MatrixXd " + i + ";"
                      ,file=f)
           
   
    with open(FileName, 'a') as f:
           print("\n"
                 "// matriz de datos de predicciones:\n"
                 "Eigen::MatrixXd Prediccion;\n"
                 "//***********************************************************\n"
                 "\n\n"
                 "int main()\n"
                 "{\n"
                 'std::ifstream fin("NumeroDatos.txt");\n'
                 "std::string name;\n"
                 "int NumeroPredicciones, VectorEntrada;\n"
                 "fin >> name >> NumeroPredicciones >> VectorEntrada ;\n"
                 'X  = openData("Entrada.csv");'
             , file=f)
           
    for i in K:
        with open(FileName, 'a') as f:
            if i[0]=="W":
                print("//Matriz capa "+str(i[1])+":\n"+
                      str(i)+' = openData("'+str(i)+'.csv");\n'
                      ,file=f) 
            else:
                print("//Sesgo capa "+str(i[1])+":\n"+
                      str(i)+' = openData("'+str(i)+'.csv");\n'
                      ,file=f)
            
    with open(FileName, 'a') as f:
           print("for(int i=0;i<=NumeroPredicciones;i++){\n"                            
                 , file=f)       
    af=1    
    for AF in Act:
        with open(FileName, 'a') as f:
            print("//Multiplicación número "+str(af)+"+ Sesgo + función de activación"                
                      ,file=f)        
            if af==1:
                print("X_W"+str(af)+"=(X(Eigen::all,Eigen::seq(Eigen::last+1-VectorEntrada,Eigen::last))*W"+str(af)+")+B"+str(af)+".transpose();\n"+
                      "X_W"+str(af)+"=X_W"+str(af)+".unaryExpr([](double x)"+Functions[AF]+"\n"
                    ,file=f)
                
            else:
                print("X_W"+str(af)+"=(X_W"+str(af-1)+"*W"+str(af)+")+B"+str(af)+".transpose();\n"+
                      "X_W"+str(af)+"=X_W"+str(af)+".unaryExpr([](double x)"+Functions[AF]+"\n"
                    ,file=f)
            af+=1   
    
    with open(FileName, 'a') as f:
           print("X.conservativeResize(X.rows(), X.cols()+1);\n"+
                 "X.col(X.cols()-1) << X_W"+str(NCapas)+".col(0);\n"+
                 "}\n"
                 "for(int i=0;i<=X.cols()-1;i++)\n"
                 "{\n"
                 "if(i<X.cols())\n"
	             "{\n"
                 "std::cout <<X(i)<<std::endl;\n"
                 "}\n"
                 "else\n"
	             "{\n"
	             "std::cout <<X(i)<<';';\n"
                 "}\n"    
                 "}\n"
                 "return 0;\n"
                 "}"                            
                 , file=f)    
#Fin de la escritura       
###################################################################################################
###################################################################################################
###################################################################################################
def ExportDataInputCMasMas(path,path_cpp,fore_seq,input_num,K,Mod):
    #Generation of input files for C++ execution
    #Input vector export:    
    Fore=fore_seq.copy()  
    Entrada = open(path+path_cpp+"Entrada.csv", "w")      
    np.savetxt(Entrada, [Fore.iloc[:int(input_num), 1].to_numpy()], delimiter=',',footer=';',comments='',newline='')  
    Entrada.close()    
    #Number of data to forecast + number of input data     
    with open(path+path_cpp+'NumeroDatos.txt', 'w') as f:
        f.write("Datos: "+str(len(fore_seq)-int(input_num))+" "+str(int(input_num)))  
        
    #Export entries (Weights + biases)
    for indice, matriz in enumerate(K):
        a_file = open(path+path_cpp+matriz+".csv", "w")
        row = Mod.get_weights()[indice]
        np.savetxt(a_file, row,delimiter=',')   
        a_file.close()
        del a_file 


def Prediction(path,path_cpp,FileName,fore_seq):
    Plataforma=platform.platform()
    Fore=fore_seq.copy()     
    cwd=os.getcwd()
    cwd2=cwd    
    os.chdir(os.path.abspath(path+path_cpp))
    
    if Plataforma.find("indows") == 1:  
        Ejecutar=FileName+".exe > Fore.dat"
    else:        
        Ejecutar="./"+FileName+" > Fore.dat"
        #This make exetutable file by user in Linux
        pl=subprocess.Popen("chmod +777 "+FileName,shell=True)    
        pl.wait()
        
    print(f"Aquí entra a Ejecutar y ejecuta: {Ejecutar}")
        
    pl=subprocess.Popen(Ejecutar,shell=True)    
    pl.wait()      
    
    ForeVector=pd.read_table("Fore.dat", header=None)      
    os.chdir(cwd2)     
    try:
        Fore.iloc[:, 1]=ForeVector
    except:
        Fore.iloc[:, 1]=ForeVector[:-1]    

    return Fore

###########################################################
######## Data Bases Functions                      ########
###########################################################
#Trend removal function
def removeTrend(error,var_model,kep_freq):

    error_comps = seasonal_decompose(error[var_model], model='additive', period=int(kep_freq))
    detr,error_trend = error.copy(),error.copy()

    #Detrending data:
    detr[var_model] = error[var_model]-error_comps._trend
    detr=detr.dropna()
    detr.index=range(0,detr.shape[0])

    #Trend component:
    error_trend[var_model]= error_comps._trend
    error_trend=error_trend.dropna()
    error_trend.index=range(0,error_trend.shape[0])

    #Error without NA's:
    errorBU=pd.concat([detr['t'], detr[var_model]+error_trend[var_model]], axis=1)

    return detr, error_trend, error_comps, errorBU

#Function that creates a basis for training, validation and testing.
def creaBases(var_order,path_file_obs,path_file_appr,resol,kep_period,var_set,input_unit,
              inputI,output_num,nfolds,sets_unit,train_set,test_set,var_model,rem_trend,
              vect_step,ResumenEjecucion,valid_set):

    ####################################################################################
    # file_obs y file_appr son los nombres de los archivos de los datos del propagador #
    # AIDA y SPG4 que se van a leer de la carpeta data.                                #
    ##################s##################################################################

    parametros={'sep':'\s+','header':None,'names':var_order}
    obs=pd.read_csv(path_file_obs, **parametros)
    appr=pd.read_csv(path_file_appr,**parametros)

    # CHECK COINCIDENCE OF TIME COLUMNS for both sets of ephemerides
    ind = obs[['t']] != appr[['t']]
    #para acceder a las filas es: obs.iloc[1:3]
    if (ind.sum()[0]!=0):
        print("Both sets of ephemerides (observed and approximate) "+
            "correspond to different instants")

    # *******************************************************************
    # FILE RESOLUTION ("t": [days], resolution: [min])

    ##############################################################################################
    # file_resol is the resolution in minutes of the data file to be processed                   #
    # file_resol The difference between the time of taking a data and the following time         #
    # Since the result is in days, it must be multiplied by 1440 to convert it to minutes.       #
    # file_resol is then used to calculate how often a line is read (sampling_period)            #
    # from the file depending on the resolution in minutes requested by the user.                #
    ##############################################################################################

    file_resol = round(((obs.loc[1,['t']] - obs.loc[0,['t']]) * 1440),2)[0]

    ########################################################################################
    # SAMPLING PERIOD Every time a row is read from the ephemeris data file.               #
    ########################################################################################
    sampling_period=int(round(resol / file_resol))

    if (sampling_period < 1):
      print("Time resolution in ephemeris files is lower than requested resolution")


    ################################################################
    # KEPLERIAN PERIOD                                             #
    # (keplerian period: [h], keplerian frequency: [file samples]) #
    #                                                              ##############################
    # The Keplerian period is the time in hours it takes for the orbiter to make one complete   #
    # revolution around the earth (14.08). kep_period == 0, is used to automatically calculate  #
    # the value of this period from the ephemeris files.                                        #
    #############################################################################################

    if (kep_period == 0):
    ###################################################################################
    # The function "switch_demo" chooses according to the selected variable to model".#
    # It is in charge of making the change of coordinates from hill.                  #
    ###################################################################################
        def switch_demo(argument):

            if argument not in ["dln", "orb", "equi", "hill", "cart"]:

                print("Unknown variable set ("+ str(var_set) + "). Possibilities: dln, orb, equi, hill, cart")

            else:

                if   argument== 'hill':
                    return coor.hill2orb(obs.copy())

                elif argument== 'cart':
                    return  coor.cart2orb(obs.copy())

                elif argument== 'dln' :
                    return  coor.dln2orb(obs.copy())

                elif argument== 'orb':
                    return obs.copy()

                else:
                    coor.equi2orb(obs.copy())

        orb = switch_demo(var_set)

    #############################################################################################
    # a2p transforms the semi-major axis to Keplerian period. This period is given in hours.    #
    # Because of this, the period is divided by 3600 to convert it to seconds.                  #
    #############################################################################################

    # kep_period = pe.a2P(orb["a"].mean()) / 3600
    kep_period=14.078234115049042
    kep_freq  = round(kep_period * 60 / file_resol)

    # *******************************************************************
    # TIME SPAN & NUMBER OF REVOLUTIONS: inputs, outputs & ephemeris files
    # (span: [h], resolution: [min], keplerian period: [h])


    # Inputs
    if (input_unit == "num"):
        input_num  = inputI
        input_span = input_num  * resol / 60
        input_rev  = input_span / kep_period
    elif (input_unit == "span"):
        input_span = inputI
        input_num  = round(input_span / (resol/ 60))
        input_rev  = input_span / kep_period


    ###################################################################################
    # input_num is the number of data that will enter the neural network, it is       #
    # calculated by multiplying the number of revolutions that will be necessary to   #
    # predict "input_rev" by the Keplerian period.                                    #
    ###################################################################################
    ###################################################################################
    # By taking the number of revolutions to be used to predict in the model and      #
    # multiplying them by the Keplerian period, you have the number of hours you need #
    # to take from the training data, that is the input_span.                         #
    ###################################################################################
    # If the input_span is divided by the resolution at which the data will be taken, #
    # calculated in hours (hence the division over 60), the number of steps that must #
    # be taken to reach that number of revolutions is obtained. That number is the    #
    # size of the input vector of the neural network.                                 #
    ###################################################################################

    elif (input_unit == "rev"):
        input_rev  = inputI
        input_span = input_rev * kep_period
        input_num  = round(input_span / (resol / 60))
    else:
      print("Wrong specification of input_unit (\""+ str(input_unit)+ "\"). "+
            "Possibilities:\n"+
            "- \"num\"  (number           )\n"+
            "- \"span\" (time        [h]  )\n"+
            "- \"rev\"  (revolutions [rev])")

    ##################################################################################
    # output_span multiplied by the resolution in hours, indicates the time needed   #
    # to take that data. In this case only 1 data =0.166666 hour.                    #
    ##################################################################################
    # Outputs
    output_span = output_num * resol / 60

    # Ephemeris files

    ####################################################################################
    # Multiplying the size of the data file, by the fraction of an hour involved in    #
    # the resolution, gives the maximum number of hours available for use in training, #
    # validating and testing the neural network. file_span is used to check that the   #
    # time required in hours needed to train, validate and test the network does not   #
    # exceed the available data time.                                                  #
    ####################################################################################

    file_span = (obs.shape[0] - 1) * file_resol / 60
    file_rev  = file_span / kep_period


    # *******************************************************************
    # AUXILIARY CALCULATIONS FOR TRAINING, VALIDATION AND TEST SETS
    # (span: [h], resolution: [min], keplerian period: [h])

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # DELIMIT SETS

    # Cross-validation -> No validation set
    if (nfolds != 0):
        valid_set=0

    # Effective spans (without overlap)

    if (sets_unit == "span"):
        train_span_eff = train_set
        valid_span_eff = valid_set
        test_span_eff  = test_set



    ####################################################################################
    # Multiplying the number of revolutions requested for validation and test training #
    # by the Keplerian period, the number of hours to be taken from the data file is   #
    # obtained. This first approach does not take into account the data needed for the #
    # first input to the model, i.e. the input_num in hours.                           #
    ####################################################################################


    elif (sets_unit== "rev"):
        train_span_eff = train_set * kep_period
        valid_span_eff = valid_set * kep_period
        test_span_eff  = test_set  * kep_period

    else:
        print("Wrong specification of sets_unit (\""+ str(sets_unit)+ "\"). "+
            "Possibilities:\n"+
            "- \"span\" (time        [h]  )\n"+
            "- \"rev\"  (revolutions [rev])")

    ################################################################################
    # Since it takes two revolutions to enter the network at the beginning of the  #
    # training, more data are needed from the training set, this amount of data is #
    # taken into account in overlap.                                               #
    ################################################################################

    # Overlap (span: [h])
    overlap  = (input_num + output_num - 1) * resol / 60

    ###################################################################################
    # The variables train_t_fin/valid_t_fin/test_t_fin store the number of hours      #
    # you will take from the data file for training, validation and test. train_t_fin #
    # contains your hours plus those needed for the first two input revolutions,      #
    # valid_t_fin contains the hours from the start of data to the end of validation. #
    # Same for test_t_fin                                                             #
    ###################################################################################

    # Final instants of sets

    train_t_fin  = overlap     + train_span_eff
    valid_t_fin  = train_t_fin + valid_span_eff
    test_t_fin   = valid_t_fin + test_span_eff
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # VERIFICATIONS

    ###############################################################################
    # This is where checks are made that the amount of data requested to train    #
    # the network will not exceed the amount of data available.                   #                                                  #
    ###############################################################################


    # ...................................................................
    # Are sets longer than the ephemeris file?

    if (train_t_fin > file_span):
      print("The required training span ("+ str(train_t_fin) + " h) "+
            "is longer than the ephemeris file ("+ str(file_span) + " h)")


    if (valid_t_fin > file_span - train_t_fin- input_span - output_span):
      print("The required span for training and validation ("+ str(valid_t_fin) + " h) "+
            "is longer than the ephemeris file ("+ str(file_span) + " h) "+
            "or does not leave enough time for one single vector in the test set ("+
            str(input_span + output_span) + " h)")


    if (test_t_fin > file_span):
      sys.stdout.write(
        "WARNING:\n"+
        "The complete required span for training, validation and test ("+
        str(round(test_t_fin,4))+" h)\n"+
        " is longer than the ephemeris file ("+ str(round(file_span,4)) +" h).\n"+
        " The effective test span ("+ str(round(test_span_eff,4))+ " h) will be reduced to "+
        str(round(file_span - valid_t_fin,4))+" h.\n\n\n")
      test_t_fin = file_span
      test_span_eff = test_t_fin - valid_t_fin

    # ...................................................................
    # ...................................................................
    # Are effective sets negative or zero?

    #################################################################################
    # The next step is to verify that there are no training, validation and test    #
    # hours with negative or zero values (except in validation in the case of cross #
    # validation nfolds!=0 ). These checks will be performed from the beginning in  #
    # later versions.                                                               #
    #################################################################################

    if (train_span_eff <= 0):
      print("The required effective training span (" + str(train_span_eff) + " h) " +
                     "is negative or zero")

    if (valid_span_eff <= 0 and  nfolds == 0):
      print("The required effective training span (" + str(valid_span_eff) + " h) " +
                     "is negative or zero")

    if (test_span_eff <= 0):
      print("The required effective training span (" + str(test_span_eff) + " h) " +
                     "is negative or zero")

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # ADDITIONAL PARAMETERS after possible corrections due to verifications

    ##################################################################################
    # As train_t_fin is given in hours, this value is passed to minutes and stored   #
    # in train_row_fin to determine how many minutes (rows) to take from the data    #
    # file for training, the same for valid_row_fin and test_row_fin. Remember that  #
    # at this point, each index has the minutes from time 0 and also, the overlap is #
    # accounted for.                                                                 #
    ##################################################################################

    # Row indices
    train_row_fin = train_t_fin * 60 / file_resol
    valid_row_fin = valid_t_fin * 60 / file_resol
    test_row_fin  = test_t_fin  * 60 / file_resol
    overlap_fin   = overlap * 60 / file_resol

    ################################################################################
    # train_span, valid_span, test_span, store the number of hours unique to each  #
    # dataset, unlike train/valid/test_t_fin which stores them from the beginning. #                                                  
    ################################################################################

    # Total spans (with overlap)

    train_span    = train_span_eff + overlap
    if (nfolds != 0):
        valid_span  = 0
    else:
        valid_span  =valid_span_eff + overlap

    test_span = test_span_eff  + overlap


    ################################################################################
    # train_rev_eff is in charge of determining the number of revolutions required #
    # to train the model. valid_rev_eff, test_rev_eff, are the revolutions for     #
    # validation and training.                                                     #
    ################################################################################


    #Number of revolutions [rev]
    train_rev_eff = train_span_eff / kep_period
    valid_rev_eff = valid_span_eff / kep_period
    test_rev_eff  = test_span_eff  / kep_period
    train_rev     = train_span     / kep_period
    valid_rev     = valid_span     / kep_period
    test_rev      = test_span      / kep_period

    # Complete interval spanned by the 3 sets (training, validation, test)
    sets_span     = test_t_fin
    sets_rev      = sets_span  / kep_period
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # *******************************************************************

    #####################################################################
    # DATA PREPARATION
    #####################################################################

    # VARIABLE TO MODEL

    ##########################################################################
    # The error is created with the difference of obs and appr data in the   #
    # variable "theta".                                                      #
    ##########################################################################

    error = pd.concat([obs['t'], obs[var_model]-appr[var_model]], axis=1)
    
    if (var_model== "theta" or var_model=="THETA2"):
        error[var_model] = an.rem2pi_sym(error[var_model])

    #############################################
    # In the event of trend removal:
    #############################################

    # When detrending, "error" is the detrended time series and "error_trend" is the trend
    errorDT, error_trend, error_comps, errorBU  = removeTrend(error,var_model,kep_freq)


    if(rem_trend):
        error = errorDT.copy()

    ###########################################################################
    # The following function creates the vectors for training and validation. #                                                   #
    ###########################################################################

    # MATRICES OF DENSE VECTORS FOR THE MODELING PROCESS
    # The error vector will be partitioned into training and validation.

    aux_vect        = ml_misc.ephem2vect(

        ephem           = error,
        column          = var_model,
        num_inputs      = input_num,
        num_outputs     = output_num,
        row_initial     = 0,
        row_final_train = int(train_row_fin),
        row_final_valid = int(valid_row_fin),
        overlap_fin     = int(overlap_fin),
        sampling_period = sampling_period,
        vect_step       = vect_step
    )

    train_vect, valid_vect = aux_vect

    # SAMPLED VECTORS FOR PLOTS (non-overlapped)
    # (column matrices)


    ###########################################################################
    # ind_train generates a vector of step numbers from sampling_period to    #
    # train_row_fin (which is in minutes). This is used to select every 10    #
    # rows, data from the error vector found above.                           #
    ###########################################################################


    ###########################################################################
    # train_row_fin gets the number of minutes (file rules) it takes to train #
    # the model. This number is passed to ind_train. For plotting, the data   #
    # needs to be sequential with step 1, so train_seq.index is used,         #
    # to remove the step of 10 that brings the vector. The same is done for   #
    # valid_seq and test_seq.                                                 #
    ###########################################################################

    ind_train       = list(np.arange(0,round(train_row_fin),sampling_period))
    train_seq       = error.loc[ind_train,:] # <-Contains the data to be plotted
    train_trend_seq = error_trend.loc[ind_train,:] # <-Contains the data to be plotted
    train_seq_num   = train_seq.shape[0]


    ############################################################################
    # The index ind_valid (below) is repeated because in list only one vector  #
    # is created from 0 to valid_row_fin with step 10, when in the second line #
    # is written: train_seq_num it is asked to read the data found from the    #
    # position train_seq_num. For example, the vector                          #
    # [0,10,20,30,40,50,50,60,70,80,90,100] is created and then taken from     #
    # position 4: [40,50,60,70,80,90,100]. It is not created as:               #
    # list(np.arange(4,round(valid_row_fin),sampling_period)) because then it  #
    # would be:[3,13,23,......103].  ind_(valid/test) has been created as      #
    #they are called later.                                                    #
    ############################################################################


    ind_valid0      = list(np.arange(0,round(valid_row_fin),sampling_period))
    ind_valid       = ind_valid0[train_seq_num:]
    valid_seq       = error.loc[ind_valid,:]
    valid_trend_seq = error_trend.loc[ind_valid,:] # Contains the data to be plotted
    valid_seq_num   = valid_seq.shape[0]


    ind_test0       = list(np.arange(0,round(test_row_fin),sampling_period))
    ind_test        = ind_test0[(train_seq_num + valid_seq_num):]
    test_seq        = error.loc[ind_test,:]
    test_trend_seq  = error_trend.loc[ind_test,:] # <-contiene los datos a graficar
    test_seq_num    = test_seq.shape[0]

    # When detrending, "x_seq" is complete (not detrended) and "x_trend_seq" is the trend
    if(rem_trend):
        train_seq[var_model]  = train_seq[var_model] + train_trend_seq[var_model]
        valid_seq[var_model]  = valid_seq[var_model] + valid_trend_seq[var_model]
        test_seq[var_model]   = test_seq[var_model]  + test_trend_seq[var_model]



    train_seq.index = range(0,train_seq_num) #indices
    valid_seq.index = range(0,valid_seq_num) #indices
    test_seq.index = range(0,test_seq_num)  #indices
#     fore_trend_seq.index =range(0,fore_trend_seq.shape[0])
    train_trend_seq.index = range(0,train_trend_seq.shape[0]) #indices
    valid_trend_seq.index = range(0,valid_trend_seq.shape[0]) #indices
    test_trend_seq.index = range(0,test_trend_seq.shape[0]) #indices

    
    with open(ResumenEjecucion) as f:
        if "PARAMETERS:             " in f.read():
            escribir=False
        else:
            escribir=True
    if escribir:
        with open(ResumenEjecucion, "a") as f:                   
            print("PARAMETERS:             "+          "              \n"+
              "------------------------------------------------------------------  \n"+
              "- Variable set:         "+str(var_set)+ "              \n"+
              "- Variable to model:    "+str(var_model)+"              \n"+
            "------------------------------------------------------------------  \n"+
              "- DETREND:              "+str(rem_trend)+"              \n"+
              "  - Frequency:          "+str(kep_freq)+" file samples \n"+
              "------------------------------------------------------------------  \n"+
              "- KEPLERIAN period:     "+str(kep_period)+" h            \n" +
            "------------------------------------------------------------------  \n"+
              "- RESOLUTION:           "+          "              \n"+
              "  - Ephemeris file:     "+str(file_resol)+" min          \n"+
              "  - Required:           "+str(resol)+   " min          \n"+
              "------------------------------------------------------------------  \n"+
            "- INPUTS:               "+          "              \n"+
              "  - Number:             "+str(input_num)+"              \n"+
              "  - Time span:          "+str(input_span)+" h            \n"+
              "  - Revolutions:        "+str(input_rev)+"              \n"+
              "------------------------------------------------------------------  \n"+
            "- OUTPUTS:              "+str(output_num)+"              \n"+
              "------------------------------------------------------------------  \n"+
            "- EPHEMERIDES in file:  "+          "              \n"+
              "  - Number:             "+str(obs.shape[0])+"              \n"+
              "  - Time span:          "+str(file_span)+" h            \n"+
              "  - Revolutions:        "+str(file_rev)+"              \n"+
              "------------------------------------------------------------------  \n"+
              "- EPHEMERIDES used:     "+          "              \n"+
              "  - Time span:          "+str(sets_span)+" h            \n"+
              "  - Revolutions:        "+str(sets_rev)+"              \n"+
              "------------------------------------------------------------------  \n"+
            "- TRAINING vectors:     "+          "              \n"+
              "  - Number:             "+str(train_seq.shape[0])+"              \n"+
              "  - TIME span:          "+          "              \n"+
              "    - Total:            "+str(train_span)+" h            \n"+
              "    - Effective:        "+str(train_span_eff)+           " h            \n"+
              "  - REVOLUTIONS:        "+          "              \n"+
              "    - Total:            "+str(train_rev)+"              \n"+
              "    - Effective:        "+str(train_rev_eff)+            "              \n"+
            "------------------------------------------------------------------  \n"+
              "- VALIDATION vectors:   "+          "              \n"+
              "  - Number:             "+str(valid_seq.shape[0])+"              \n"+
              "  - TIME span:          "+          "              \n"+
              "    - Total:            "+str(valid_span)+" h            \n"+
              "    - Effective:        "+str(valid_span_eff),           " h            \n"+
              "  - REVOLUTIONS:        "+          "              \n"+
              "    - Total:            "+str(valid_rev)+"              \n"+
              "    - Effective:        "+str(valid_rev_eff),            "              \n"+
              "------------------------------------------------------------------  \n" +
            "- TEST vectors:         "+          "              \n"+
              "  - Number:             "+str(test_seq.shape[0])+"              \n"+
              "  - TIME span:          "+          "              \n"+
              "    - Total:            "+str(test_span)+" h            \n"+
              "    - Effective:        "+str(test_span_eff),            " h            \n"+
              "  - REVOLUTIONS:        "+          "              \n"+
              "    - Total:            "+str(test_rev)+"              \n"+
              "    - Effective:        "+str(test_rev_eff)+"              \n"+
              "------------------------------------------------------------------  \n"
              , file=f)


    return (ind_train,train_seq,train_seq_num,train_trend_seq,train_vect,ind_valid,ind_valid0,valid_seq,valid_seq_num,valid_trend_seq,valid_vect,ind_test,ind_test0,test_seq,test_seq_num,test_trend_seq,appr,obs,error,error_comps,error_trend,errorBU,input_num)

##############################################################################################################
###### Function to create fore sets to forecasting
##############################################################################################################

def Select_fore_set(fore_set,ind_train,nfolds,ind_valid0,train_seq_num,input_num,output_num,
                    ind_test0,valid_seq_num,error,errorBU,error_trend,ind_valid,ind_test,obs,appr,var_model):
    # SAMPLED VECTOR FOR FORECASTING (partially overlapped with valid span)
    # (column matrix)
    ############################################################################
    # As two input revolutions are needed when predicting in training, the     #
    # overlap should be left. But to predict in validation and test the two    #
    # input revolutions of the previous dataset are used, so those revolutions #
    # are removed ( - (int(input_num))                                         #
    ############################################################################

    if (fore_set == "train"):
        ind_fore = ind_train

    ############################################################################
    # ind_fore when in validation or testing, is found by subtracting the      #
    # input_num from the validation or test file size.                         #
    ############################################################################

    elif (fore_set == "valid" and nfolds == 0):

        ind_fore = ind_valid0[(train_seq_num-(int(input_num)+output_num - 1)):]



    elif (fore_set == "test"):

        ind_fore = ind_test0[((train_seq_num + valid_seq_num) - (int(input_num) + output_num - 1)):]



    elif (fore_set == 'train-valid' and nfolds == 0):

        ind_fore   = ind_valid0



    elif (fore_set == 'valid-test' and nfolds == 0):

        ind_fore = ind_test0[(train_seq_num - (int(input_num) + output_num - 1)):]



    elif (fore_set == "train-valid-test" and nfolds == 0):

        ind_fore   = ind_test0

    elif (fore_set == "train-test"):

        ind_fore   = ind_test0
        

    else:
      print("Wrong specification of set to be forecasted (\""+ str(fore_set) + "\"). " +
            "Possibilities:\n" +
            "- \"test\"  (usual)\n" +
             "- \"train\" (trials)\n" +
              "- \"valid\" (trials with no Cross-validation)" )

    # When detrending, "fore_seq" is initially detrended for the forecast process
    # (only the starting section, the rest is 0).
    # After forecasting, the trend ("fore_trend_seq") will be added

    fore_seq       = error.loc[ind_fore,:]
    fore_trend_seq = error_trend.loc[ind_fore,:]
    fore_seq_num   = fore_seq.shape[0]
    fore_seq.index = range(0,fore_seq_num)
    Reales         = fore_seq.copy()
    Reales_trend   = errorBU.loc[ind_fore,:]
    Reales_trend.index = range(0,Reales_trend.shape[0])
    fore_seq.loc[list(np.arange(int(input_num),fore_seq_num)),fore_seq.columns.values[1]:] = 0
    # *******************************************************************


#     # *******************************************************************
#     # SAMPLED EPHEMERIDES FOR FINAL HYBRID PROPAGATION AND POSITION ERROR (non-overlapped)
#     # (2D matrices)

    if (fore_set == "train"):
        ind_aux = ind_train[(int(input_num) + output_num - 1):]

    elif (fore_set == "valid"):

        ind_aux = ind_valid

    elif (fore_set == "test"):
        ind_aux = ind_test


    elif (fore_set == 'train-valid'):

        ind_aux = ind_valid0[(int(input_num) + output_num - 1):]


    elif (fore_set =='valid-test'):

        ind_aux = ind_test0[(int(input_num)+train_seq_num + output_num- 1)::]


    elif (fore_set == 'train-valid-test'):

        ind_aux   = ind_test0[(int(input_num) + output_num - 1):]

    elif (fore_set == 'train-test'):

        ind_aux   = ind_test0[(int(input_num) + output_num - 1):]

    else:
        print("Wrong specification of set to be forecasted (\""+ str(fore_set) + "\"). "+
            "Possibilities:\n"+
            "- \"test\"  (usual)\n"+
             "- \"train\" (trials)\n"+
              "- \"valid\" (trials with no Cross-validation)")
    obs_test  = obs.loc[ind_aux,:]
    appr_test = appr.loc[ind_aux,:]


    # When detrending, "fore_test" is initially empty.
    # After forecasting, it will contain the complete forecast (not detrended).
    # "fore_trend_test" contains the trend
    fore_test = obs_test.copy()
    fore_test.iloc[:, fore_test.columns != 't']=0

    fore_trend_test = fore_test.copy()
    fore_trend_test[var_model] = error_trend.loc[ind_aux, var_model]



    fore_test.index = range(0,fore_test.shape[0])
    obs_test.index = range(0,obs_test.shape[0])
    appr_test.index = range(0,appr_test.shape[0])
    fore_trend_seq.index =range(0,fore_trend_seq.shape[0])

    return fore_seq,fore_test,obs_test,appr_test,fore_trend_seq,Reales,Reales_trend

##############################################################################################################
###### Functions for plotting
##############################################################################################################
#Basic Plot#
def BasicPlot(var_model,fileplot,plot_width,plot_height,train_seq,input_num,output_num,
          train_seq_num,test_seq,nfolds,valid_seq,var_lab_unit):   

    greek_letterz=[chr(code) for code in range(945,970)]   #letra theta
    if var_model=="theta" or var_model=="THETA2":
        var_lab_name=greek_letterz[7]
    else:
        var_lab_name=var_model

    with PdfPages(fileplot) as export_pdf:

        #Input
        pyplot.figure(figsize=(plot_width,plot_height))
        pyplot.plot(train_seq.loc[0:int(input_num +output_num- 2),'t'],
        train_seq.loc[0:int(input_num +output_num- 2),var_model], ':',
        #train
        train_seq.loc[(input_num + output_num - 2):train_seq_num,'t'],
        train_seq.loc[(input_num + output_num - 2):train_seq_num,var_model],'g',
        #test
        test_seq.t, test_seq[var_model],'b')
        #valid
        if(nfolds==0):
            pyplot.plot(valid_seq.t, valid_seq[var_model],'r')
        #ylabel
        pyplot.ticklabel_format(axis='y',style='sci',scilimits=(1,4), useMathText=True)      
        pyplot.rc('font', size=15)
        #ylabel
        pyplot.ylabel(var_lab_name + " error (" + var_lab_unit + ")",
                      fontsize=15)
        pyplot.yticks(fontsize=15)
        #xlabel
        pyplot.xlabel("t (days)",fontsize=15)
        pyplot.xticks(fontsize=15)
        pyplot.tight_layout()
        export_pdf.savefig(dpi=600)
        pyplot.close()
#Plot if detrending#
def plot_decomp(fileplotDetrending,error_comps):
    with PdfPages(fileplotDetrending) as export_pdf:
        
        original= error_comps._observed
        trend = error_comps.trend
        seasonal = error_comps.seasonal
        residual = error_comps.resid
        
        
        fig = pyplot.figure()
        gs = fig.add_gridspec(4, hspace=0.4)
        axs = gs.subplots(sharex=True, sharey=False)
        
        axs[0].plot(original,'tab:orange')
        axs[0].set_title('Original', fontsize=10)
        axs[0].ticklabel_format(axis='y',style='sci',scilimits=(1,4), useMathText=True)
        axs[0].ticklabel_format(useMathText=True)
        axs[0].tick_params(axis='both', which='both', labelsize=10)
        axs[0].xaxis.set_ticks_position('none')
        
        
        axs[1].plot(trend,'tab:green')
        axs[1].set_title('Trend', fontsize=10)
        axs[1].ticklabel_format(axis='y',style='sci',scilimits=(1,4), useMathText=True)
        axs[1].ticklabel_format(useMathText=True)
        axs[1].tick_params(axis='both', which='both', labelsize=10)
        axs[1].xaxis.set_ticks_position('none')

        axs[2].plot(seasonal,'tab:red')
        axs[2].set_title('Seasonality', fontsize=10)
        axs[2].ticklabel_format(axis='y',style='sci',scilimits=(1,4), useMathText=True)
        axs[2].ticklabel_format(useMathText=True)
        axs[2].tick_params(axis='both', which='both', labelsize=10)
        axs[2].xaxis.set_ticks_position('none')

        axs[3].plot(residual)
        axs[3].set_title('Residuals', fontsize=10)
        axs[3].ticklabel_format(axis='y',style='sci',scilimits=(1,4), useMathText=True)
        axs[3].ticklabel_format(useMathText=True)
        axs[3].tick_params(axis='both', which='both', labelsize=10)  
        # axs[3].xaxis.set_ticks_position('none')
        
        pyplot.rc('font', size=10)
        
        export_pdf.savefig(dpi=300)
        pyplot.close()
        
def plot_complete(fileplotComplete,errorBU,var_model,error_trend,errorDT,var_lab_unit):
    greek_letterz=[chr(code) for code in range(945,970)]   #letra theta
    if var_model=="theta" or var_model=="THETA2":
        var_lab_name=greek_letterz[7]
    else:
        var_lab_name=var_model
    with PdfPages(fileplotComplete) as export_pdf:
        pyplot.plot(errorBU.t,errorBU[var_model], 'm'
        ,errorBU.t,error_trend[var_model],'b',
        errorBU.t,errorDT[var_model],'g')
        pyplot.ticklabel_format(axis='y',style='sci',scilimits=(1,4), useMathText=True)      
        pyplot.rc('font', size=15)
        #ylabel
        pyplot.ylabel(var_lab_name + " error (" + var_lab_unit + ")",
                      fontsize=15)
        pyplot.yticks(fontsize=15)
        #xlabel
        pyplot.xlabel("t (days)",fontsize=15)
        pyplot.xticks(fontsize=15)
        pyplot.tight_layout()
        export_pdf.savefig(dpi=300)
        pyplot.close()
#Forecasting plot:
def plot_fore(fore_seq,reales,nombregrafico,plot_width,plot_height,input_num,
              output_num,var_model,var_lab_unit):
    # Forecast
    fore_seq_plot = fore_seq[:]
    fore_seq_num=fore_seq_plot.shape[0]
    greek_letterz=[chr(code) for code in range(945,970)]   #letra theta
    if var_model=="theta" or var_model=="THETA2":
        var_lab_name=greek_letterz[7]
    else:
        var_lab_name=var_model

    with PdfPages(nombregrafico +'.pdf') as export_pdf:

      #Input
      #pyplot.style.use('dark_background')
      pyplot.figure(figsize=(plot_width,plot_height))
      pyplot.plot(reales.loc[0:int(input_num +output_num- 2),'t'],
      reales.loc[0:int(input_num +output_num- 2),var_model], ':',
      #Reals
      reales.loc[(input_num + output_num - 2):,'t'],
      reales.loc[(input_num + output_num - 2):,var_model],"b",
      #Fore
      fore_seq_plot.loc[(input_num + output_num):fore_seq_num, "t"],
      fore_seq_plot.loc[(input_num + output_num):fore_seq_num,var_model],'m')
      #ylabel
      pyplot.ticklabel_format(axis='y',style='sci',scilimits=(1,4), useMathText=True)      
      pyplot.rc('font', size=15)
      #ylabel
      pyplot.ylabel(var_lab_name + " error (" + var_lab_unit + ")",
                    fontsize=15)
      pyplot.yticks(fontsize=15)
      #xlabel
      pyplot.xlabel("t (days)",fontsize=15)
      pyplot.xticks(fontsize=15)
      pyplot.tight_layout()
      export_pdf.savefig(dpi=600)
      pyplot.close()
##############################################################################################################
###### Function to loss Plot
###############################################################################################################
def LossPlot(pathLostPlot,history,lista):
    pyplot.clf()
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('Model loss')    
    pyplot.legend(['train', 'test'], loc='best')
    pyplot.ylabel("Loss",fontsize=15)
    pyplot.yticks(fontsize=15)
    #xlabel
    pyplot.xlabel("Epoch",fontsize=15)
    pyplot.xticks(fontsize=15)
    pyplot.ticklabel_format(axis='y',style='sci',scilimits=(1,4), useMathText=True)      
    pyplot.rc('font', size=15)
    pyplot.tight_layout()
    pyplot.savefig(pathLostPlot+"loss"+str(lista)+".pdf") 
###########################################################
######## DeepLerning Model hyper-paremeters Functions   ###
###########################################################

def DeepLerningModelConfigs(Parametros):   
    I=time.time()
    Configuraciones=[]  
    ListPar=[]  
    Variables=[]
    ParametrosA=Parametros.copy()    
    hl=Parametros['HiddenLayers']
    ParametrosA.pop('HiddenLayers', None)
    ParametrosA.pop('FuncAct', None)


    for i in  hl:  
        for h in range(i):   
            ParametrosA['FuncAct'+str(h)]=Parametros['FuncAct']
            
        tempVar2 = 'Configuraciones'+str(i)
        valores=list(ParametrosA.values())
        valores.append([i])
        variables=list(ParametrosA.keys())
        variables.append('HiddenLayers')
        ListPar.append(variables)
        globals()[tempVar2]=list(itertools.product(*valores))
        var=list(itertools.repeat(variables, len(globals()[tempVar2])))
        Variables+=var
        Configuraciones+=globals()[tempVar2]
        
    hp_all=[{k: v for k, v in zip(lista2[0], lista2[1])} for lista2 in zip(Variables,Configuraciones)]    
    print("DeepLerningModelConfigs function time: ", str(round(time.time()-I,2))+'\n')
    return hp_all


###########################################################
######## Deep Learning Functions                   ########
###########################################################

###############################################################################################################
###### Adjusts model under the hyper-parameters received in "LISTA".
###############################################################################################################   
def OPtimizadoryMetrica(OPT,LR):
######################################################################################################
# A vector with the possible optimizers is defined to evaluate the cost function of the neural network
######################################################################################################
    opts=['Adam', "Rmsprop","Adadelta", "Nadam"]

    optimizadores=[tf.keras.optimizers.Adam(lr=LR),
                   tf.keras.optimizers.RMSprop(lr=LR),
                   tf.keras.optimizers.Adadelta(lr=LR),
                   tf.keras.optimizers.Nadam(lr=LR) ]
    return  optimizadores[opts.index(OPT)]

#Create the "model shell" by using the Tensorflow library.
def create_model(HiddenLayers,windows,NumNeurons, loss, opt, metric, lista,fachl, LR):
    
    model = tf.keras.models.Sequential()
###############################################################################################################
###### initializing the weights
###############################################################################################################

    initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="normal", seed=None)
#######################################################################
# The variables that display the "Undefined" error are created using  #
# the "for" function above, which creates each variable by assigning  #
# values using the globals()[temVar1] value.                          #
#######################################################################

    if HiddenLayers==0:
        # If there are 0 hidden layers, only the output layer is created.
        model.add(tf.keras.layers.Dense(1,activation="linear", input_shape=(windows,), name="H-Layer",kernel_initializer=initializer))
    else:
        # fachl: Growth-decrease factor in the number of neurons from the second
        # hidden layer onward.
        act=1 #"act" gives name to the hidden layer
        Entrada=NumNeurons
        for i in range(HiddenLayers):
            if i==0:
                # First Hidden Layer
                model.add(tf.keras.layers.Dense(Entrada,activation=lista["FuncAct"+str(i)], input_shape=(windows,), name="H-Layer_1",kernel_initializer=initializer))
            else:
                #Remaining hidden layers
                Entrada=int(Entrada*fachl)
                model.add(tf.keras.layers.Dense(Entrada,activation=lista["FuncAct"+str(i)], name='H-Layer_'+str(act+1)))
                act=act+1
        # Output Layer
        model.add(tf.keras.layers.Dense(1,activation="linear", name="Output-Layer"))
###############################################################################################################
###### Compiling the neural network in Keras
###############################################################################################################
    optimizer = OPtimizadoryMetrica(opt,LR)
    model.compile(loss=loss,optimizer=optimizer,metrics = metric)
    model.summary()
    model.get_config()
    return model

###############################################################################################################
###### Model Fit
###############################################################################################################


def model_fit(path,lista,input_num,fachl,stop_tolerance,
              parametros_de_entrada,epocas, nfolds, train_vect,
              verbose):
###############################################################################################################
####### LOST FUNCTION
###############################################################################################################
    metric  = ['mse', 'mae', 'mape','msle']
    model=create_model(lista["HiddenLayers"],int(input_num),lista["NumNeurons"], lista["loss"], lista["opt"], metric, lista,fachl, lista["LR"])
###############################################################################################################
######   Early stopping
###############################################################################################################
    Monitor = EarlyStopping(monitor=lista["monitor"], min_delta=0, patience=stop_tolerance, verbose=verbose, mode='auto', restore_best_weights=True)
###############################################################################################################
######  Model training
###############################################################################################################
    if nfolds==0:
        history=model.fit(parametros_de_entrada[0],
    parametros_de_entrada[1],
    validation_data =(parametros_de_entrada[2],
    parametros_de_entrada[3]),
    batch_size=lista["BachSize"],
    epochs=epocas, verbose=verbose,
    callbacks=[Monitor])        
    else:
        X=parametros_de_entrada[0]
        y=parametros_de_entrada[1]
        kf = KFold(n_splits=nfolds)
        kf.get_n_splits(X)
        KFold(n_splits=nfolds, random_state=None, shuffle=False)        
        i=1
        for train_index, test_index in kf.split(train_vect):
            print("FOLD: ", str(i), "\n")                       
            history=model.fit(X[train_index],
            y[train_index],
            validation_data =(X[test_index] ,
            y[test_index]),
            batch_size=lista["BachSize"],
            epochs=epocas, verbose=verbose,
            callbacks=[Monitor])
            i=i+1
    Epoca = Monitor.stopped_epoch
    error = history.history[lista["monitor"]][Epoca]
    
    return model,error, Epoca, history
###############################################################################################################
##### Function that repeats the process of creating the model with the same
##### architecture to take the average value.
###############################################################################################################
def modelConstruct_R(n_repeats,lista,input_num,fachl,
                     stop_tolerance,parametros_de_entrada,epocas,
                     nfolds,path,path_plot, train_vect,verbose,
                     var_order,path_file_obs,
                     path_file_appr,resol,kep_period,var_set,
                     input_unit,inputI,output_num,sets_unit,
                     train_set,valid_set,test_set,var_model,
                     var_angular,rem_trend,vect_step,
                     ResumenEjecucion,t_dist,m,
                     PathEIGEN,Functions,path_cpp):  
    
    
       
    Inicio, InmodelConstruct_R=time.time(),time.time()
    Bases=creaBases(var_order,path_file_obs,path_file_appr,resol,kep_period,var_set,input_unit,
                  inputI,output_num,nfolds,sets_unit,train_set,test_set,var_model,rem_trend,
                  vect_step,ResumenEjecucion,valid_set)

    ind_train=Bases[0]
    train_seq_num=Bases[2]
    train_vect=Bases[4]
    ind_valid=Bases[5]
    ind_valid0=Bases[6]
    valid_seq_num=Bases[8]
    ind_test=Bases[11]
    ind_test0=Bases[12]
    appr=Bases[16]
    obs=Bases[17]
    error=Bases[18]
    error_trend=Bases[20]
    errorBU=Bases[21]
    input_num=Bases[22]   
    
    prt=str("#############################################################################\n"+
          "Function modelConstruct_R model number "+str(m)+"\n"
          "Parameter List: "+ str(lista) + "\n"+
          "The time CREATING TRAINING BASES is: "+ str(time.time()-Inicio)+ " seconds \n"+
          "#############################################################################\n")   
    with open(ResumenEjecucion, 'a') as f:
        print(prt
          , file=f)   
    print(prt)
    Inicio=time.time()
    fore_seq,fore_test,obs_test,appr_test,fore_trend_seq,Reales,Reales_trend=Select_fore_set("train",
    ind_train,nfolds,ind_valid0,train_seq_num,input_num,output_num,
    ind_test0,valid_seq_num,error,errorBU,error_trend,ind_valid,
    ind_test,obs,appr,var_model)
    
    prt=str("#############################################################################\n"+
          "The time FOR ESTABLISHING FORECASTING BASES is: "+ str(time.time()-Inicio)+ " seconds \n"+
          "#############################################################################\n")
    with open(ResumenEjecucion, 'a') as f:    
        print(prt
          , file=f) 
    print(prt)
    tiempos = list()
    error_scores = list()
    error_Days = list()
    models = list()
    epochs = list()
    historys = list()
    FileCppNames=list()
    FileEjectNames=list()
    Cols = [str(n)+'d' for n in t_dist]    
    for r in range(n_repeats):
        FileName="traincpp"+str(r)
        InRep=time.time()
        prt=str("#############################################################################\n"+
          "Repetition "+str(r)+"\n"+
          "...\n"+
          "#############################################################################\n")
        with open(ResumenEjecucion, 'a') as f:
            print(prt
              , file=f)
        print(prt)    
        # fit and evaluate the model n times   
        Inicio=time.time()
        model,scores, epoch, history = model_fit(path,lista,input_num,fachl,stop_tolerance,parametros_de_entrada,epocas,nfolds,train_vect,verbose)       
        prt=str("#############################################################################\n"
              "The time MODEL CREATION IS: "+ str(time.time()-Inicio)+ " seconds \n"+
              "#############################################################################\n")
        with open(ResumenEjecucion, 'a') as f:        
            print(prt
              , file=f) 
        print(prt)    
        

        #######################################################################################################
        #######################################################################################################
        #############   EXTRACTING INFORMATION FOR C++ FORECASTING
        #######################################################################################################
        #######################################################################################################
        
        #Crea archivo LeerPesosdetxt.h  
        Inicio=time.time()
        if not os.path.isfile(path+path_cpp+"LeerPesosdetxt.h"):
            LeerPesosdetxt(path,path_cpp)
        
        #Extract Model information
        NCapas,Act,K=ExtractModInfo(model,ResumenEjecucion)        
    
        #Export input files for C++ execution
        ExportDataInputCMasMas(path,path_cpp,fore_seq,input_num,K,model)
        
        ######################################
        #### Crea archivo RedNeuronal.cpp
        if not os.path.isfile(path+path_cpp+FileName+".cpp"):  
            creaRedNeuronalcMasMas(path+path_cpp+FileName+".cpp",K,Act,NCapas,Functions)
        ######################################
            
        ######################################
        #### Compile RedNeuronal.cpp file 
        Inicio=time.time()
        Plataforma=platform.platform()
        
        p=os.path.abspath(PathEIGEN)
        
        if Plataforma.find("indows") == 1:
            FileVerification=FileName+".exe"
        else:
            FileVerification=FileName
        print(f"FileVerification: {FileVerification}")
        if not os.path.isfile(path+path_cpp+FileVerification):
            cwd=os.getcwd()
            cwd2=cwd                
            os.chdir(os.path.abspath(path+path_cpp))
            
            #O1, O2 or O3 can be used. As it increases it loses accuracy. 
            if Plataforma.find("indows") == 1:
                run=f'g++ -O2 -I "{p}'+' '+FileName+ '.cpp -o '+FileVerification
            else:
                # run=f'g++ -std=c++0x -O2 -I "{p}"'+' '+FileName+ '.cpp -o '+FileVerification
                run=f'g++ {FileName}.cpp -o {FileVerification}'
            print(f"Run:\n{run}")
            pl=subprocess.Popen(run,shell=True) 
            print(f"Ejecución: {pl.wait()}")
            print("Compiling...")
            os.chdir(cwd2) 
            prt=str("#############################################################################\n"+
                  "The C++ COMPILATION TIME OF THE MODEL IS: "+ str(time.time()-Inicio)+ " seconds \n"+
                  "#############################################################################\n")
            with open(ResumenEjecucion, 'a') as f:        
                print(prt
                  , file=f) 
            print(prt)    
        else:
            print("Do not compile")               
              
        Inicio=time.time()    
        DistKMhyb=forec(t_dist,fore_seq,obs_test,appr_test, model,fore_test,"",input_num,
                  output_num,var_model,var_angular,var_set,FileName,Functions,
                  path,path_cpp)[2]
        prt=str("#############################################################################\n"+
              "The time CREATING DistKMhyb: "+ str(time.time()-Inicio)+ " seconds \n"+
              "#############################################################################\n")
        with open(ResumenEjecucion, 'a') as f:        
            print(prt
              , file=f)         
        print(prt)
        DistKMhyb[DistKMhyb == 0] = np.nan
        AvgDays = DistKMhyb[Cols].mean(axis=1,skipna = True)
        error_scores.append(scores)
        error_Days.append(AvgDays.values[0])
        models.append(model)
        epochs.append(epoch)
        historys.append(history)
        FileCppNames.append(FileName)
        FileEjectNames.append(FileVerification)
        tiempos.append(round(time.time()-InRep,2))
        prt=str("The total repetition time "+str(r)+" es: "+ str(time.time()-InRep)+ " seconds \n"+
              "The error is: "+ str(scores)+"\n"+
              "The average error in distance is: "+str(AvgDays.values[0])+"\n"
              "The number of epochs of this repetition: "+str(epoch)+"\n"+
              "#############################################################################\n"+
              "#############################################################################\n")
        with open(ResumenEjecucion, 'a') as f: 
            print(prt
              , file=f)
        print(prt)

    # summarize score
    errorAvgDays  = min(error_Days) #The average error is selected.
    near          = error_Days.index(errorAvgDays) #Choose the model whose error is closest to the average.
    modelo        = models[near]
    epoca         = epochs[near]
    error         = error_scores[near]
    FileName      = FileCppNames[near]
    FileVerification=FileEjectNames[near]
    
    print("The best repetition was: "+str(near))  
   
    # modelo.summary()
    LossPlot(path+path_plot+"loss_",historys[near],str(m))
    

    filesHere=os.listdir(path+path_cpp)
    for f in filesHere:
        if not (f==FileVerification or f=="EINGEN"):
            os.remove(path+path_cpp+f)


       
    prt=str("#############################################################################\n"+
          "General Summary: modelConstruct_R function model number "+str(m)+"\n"+
          "#############################################################################\n"+
          "The total time to create the model with "+str(r+1)+" repetitions is: "+ str(time.time()-InmodelConstruct_R)+ " seconds \n"+
          "The best error is: "+ str(error)+"\n"+
          "The smallest average error in distance is: "+str(errorAvgDays)+"\n"
          "The number of epochs of the best model is: "+str(epoca)+"\n"+
          "Repetition of the best model was the number "+str(near)+"\n"+
          "#############################################################################\n"+
          "#############################################################################\n")    
    with open(ResumenEjecucion, 'a') as f:
        print(prt
          , file=f)  
    print(prt)
    return error, epoca, modelo,errorAvgDays, FileVerification
    

##### FORECASTING FUNCTIONS:
##### Deep learning

def forec(t_dist,fore_seq,obs_test,appr_test, Mod,fore_test,m,input_num,
          output_num,var_model,var_angular,var_set,FileName,Functions,
          path,path_cpp):
    
    Cols = [str(n)+'d' for n in t_dist] 
    DistKMAppr,DistKMApprBest,DistKMhyb=pd.DataFrame(columns=Cols),pd.DataFrame(columns=Cols),pd.DataFrame(columns=Cols)    
    dataPredic = fore_seq.copy()
    obs_testC= obs_test.copy()
    appr_testC= appr_test.copy()
    
    appr_testBest= appr_test.copy()
    thetaAIDA=obs_testC['theta']
    appr_testBest["theta"]=thetaAIDA
    
##############################################################################################################
######Uso de función Predition
##############################################################################################################
    T_prediccionUnModelo=time.time()
    Fore=Prediction(path,path_cpp,FileName,dataPredic)       
    F_prediccionUnModelo=time.time()
    T_Ejecucion_prediccionUnModelo = round(F_prediccionUnModelo - T_prediccionUnModelo,3)
    print("The forecast time of the model is:",str(T_Ejecucion_prediccionUnModelo), " seconds"+'\n')           
    
    
    val=Fore.loc[range(int(input_num + output_num-1),fore_seq.shape[0]), [var_model]]
    val.index=range(0,len(val))
    fore_test[var_model] = val
    fore_testC= fore_test.copy()   
    
    
    def hyb_distFun(argument):
        if argument not in ["dln", "orb", "equi", "hill", "cart"]:
            print("Unknown variable set ("+ str(var_set) + "). Possibilities: dln, orb, equi, hill, cart")
        else:
            if   argument== 'hill':
                return coor.hill2dist(obs_testC, coor.hill_anal2hyb(appr_testC, fore_testC))
            elif argument== 'cart':
                return  coor.cart2dist(obs_testC, coor.cart_anal2hyb(appr_testC, fore_testC))
            elif argument== 'dln' :
                return  coor.dln2dist(obs_testC, coor.dln_anal2hyb(appr_testC,fore_testC ))
            elif argument== 'orb':
                return coor.orb2dist(obs_testC, coor.orb_anal2hyb(appr_testC, fore_testC))
            else:
                coor.equi2dist(obs_testC, coor.equi_anal2hyb(appr_testC, fore_testC))
                
    hyb_dist = hyb_distFun(var_set)  
    
    def appr_distFun(argument):
        if argument not in ["dln", "orb", "equi", "hill", "cart"]:
            print("Unknown variable set ("+ var_set + "). Possibilities: dln, orb, equi, hill, cart")

        else:
            if   argument== 'hill':
                return coor.hill2dist(obs_testC, appr_testC)                
            elif argument== 'cart':
                return  coor.cart2dist(obs_testC, appr_testC)     
            elif argument== 'dln' :
                return  coor.dln2dist( obs_testC, appr_testC)                
            elif argument== 'orb':
                return coor.orb2dist( obs_testC, appr_testC)                
            else:
                coor.equi2dist(obs_testC, appr_testC)
            
    appr_dist = appr_distFun(var_set)
    
    def appr_distFunBest(argument):
        if argument not in ["dln", "orb", "equi", "hill", "cart"]:
            print("Unknown variable set ("+ var_set + "). Possibilities: dln, orb, equi, hill, cart")

        else:
            if   argument== 'hill':
                return coor.hill2dist(obs_testC, appr_testBest)                
            elif argument== 'cart':
                return  coor.cart2dist(obs_testC, appr_testBest)     
            elif argument== 'dln' :
                return  coor.dln2dist( obs_testC, appr_testBest)                
            elif argument== 'orb':
                return coor.orb2dist( obs_testC, appr_testBest)                
            else:
                coor.equi2dist(obs_testC, appr_testBest)
            
    appr_distBest = appr_distFunBest(var_set)    
    
##############################################################################################################
###### MAX POSITION ERRORS (INSTANTS)
##############################################################################################################
    
    # Array for max position errors (abs values): _(t) x 2(appr,hyb) x 4(dist_max)
    
    dist_max_appr=pd.DataFrame(index = Cols,columns=['dis','alo','cro','rad'])
    dist_max_apprBest=pd.DataFrame(index = Cols,columns=['dis','alo','cro','rad'])
    dist_max_hyb=pd.DataFrame(index = Cols,columns=['dis','alo','cro','rad'])
       
    # Shifted "t" to start from 0
    t_ref0 = obs_test["t"] - obs_test["t"][0]
    
    # Matrix of instants for calculating max position error
    t_inst = np.asarray(t_dist)
    
    def max_finder(t_inst, mat_dist):
        a=list(abs(t_ref0 - t_inst))
        ind = a.index(min(a))
        if (max(t_ref0) >= t_inst):
            X= abs(mat_dist.loc[list(np.arange(0,ind+1)), mat_dist.columns != 't'])
            maxi= X.apply(lambda x : max(x))
        else:
            maxi= pd.Series([0.0,0.0,0.0,0.0],index=['dis','alo','cro','rad']) 
    
        return  maxi['dis'], maxi['alo'] , maxi['cro']  ,maxi['rad'] 
    
    
    for index, row in pd.DataFrame(t_inst).iterrows():
        row=int(row)
        dist_max_appr['dis'][index]=max_finder(row,appr_dist)[0]
        dist_max_apprBest['dis'][index]=max_finder(row,appr_distBest)[0]        
        dist_max_hyb['dis'][index]=max_finder(row,hyb_dist)[0]
        dist_max_appr['alo'][index]=max_finder(row,appr_dist)[1]
        dist_max_apprBest['alo'][index]=max_finder(row,appr_distBest)[1]        
        dist_max_hyb['alo'][index]=max_finder(row,hyb_dist)[1]
        dist_max_appr['cro'][index]=max_finder(row,appr_dist)[2]
        dist_max_apprBest['cro'][index]=max_finder(row,appr_distBest)[2]        
        dist_max_hyb['cro'][index]=max_finder(row,hyb_dist)[2]
        dist_max_appr['rad'][index]=max_finder(row,appr_dist)[3]
        dist_max_apprBest['rad'][index]=max_finder(row,appr_distBest)[3]        
        dist_max_hyb['rad'][index]=max_finder(row,hyb_dist)[3]

##############################################################################################################
###### Results:
##############################################################################################################
    #Approximado   
    a=pd.DataFrame(dist_max_appr["dis"])
    a.rename(columns={'dis': m}, inplace=True)   
    DistKMAppr = DistKMAppr.append(a.transpose())  

    #Híbirdo
    ah=pd.DataFrame(dist_max_hyb["dis"])
    ah.rename(columns={'dis': m}, inplace=True)    
    DistKMhyb = DistKMhyb.append(ah.transpose())      
    
    #BestApproximado   
    c=pd.DataFrame(dist_max_apprBest["dis"])
    c.rename(columns={'dis': m}, inplace=True) 
    DistKMApprBest = DistKMApprBest.append(c.transpose())

    return DistKMAppr,DistKMApprBest,DistKMhyb,Fore,Cols

##############################################################################################################
###### Function that generates each model and stores both the model and its error values information.
##############################################################################################################


def KerasSerie(train_vect,valid_vect,fileCandidato,path_path_models,
               n_repeats,lista,input_num,fachl,stop_tolerance,epocas,
              nfolds,path,path_plot,verbose,var_order,path_file_obs,
              path_file_appr,resol,kep_period,var_set,input_unit,inputI,
              output_num,sets_unit,train_set,valid_set,test_set,var_model,
              var_angular,rem_trend,vect_step,ResumenEjecucion,t_dist,
              PathEIGEN,Functions,path_cpp):

##############################################################################################################
######  Selecting training and validation data
##############################################################################################################

    parametros_de_entrada=[]
##############################################################################################################
###### Constructing the training vectors: X_train and y_train.
##############################################################################################################

    parametros_de_entrada.append(train_vect[:, :-1])
    parametros_de_entrada.append(train_vect[:,  -1])

##############################################################################################################
######  Constructing the validation vectors: X_valid and y_valid.
##############################################################################################################

    parametros_de_entrada.append(valid_vect[:, :-1])
    parametros_de_entrada.append(valid_vect[:,-1])


    ColumnsCandidatos=["Nombre",'Candidato', 'Error', 'errorAvgDays','Epochs', 'time']

    #Validates if the "Candidatos" file already exists
    if os.path.isfile(fileCandidato):
        CandidatosSave=pd.read_csv(fileCandidato, index_col=0)
        if str(lista) in list(CandidatosSave["Candidato"]):
            m=CandidatosSave.index[CandidatosSave["Candidato"]==str(lista)].values[0]   
        else:
            m=list(CandidatosSave.index)[-1:][0]+1
    else:
        CandidatosSave=pd.DataFrame(columns=ColumnsCandidatos)
        m=0


##############################################################################################################
###### Creating model With Cpp
##############################################################################################################
    
    #Using modelConstruct_R function           
    Nombre="modelo"+str(m)+".h5"    
    NombreCMasMas="modelo"+str(m)
    if os.path.isfile(path_path_models+Nombre):
        print("Model "+str(m)+" has already been created")        
    else:
        print("Creating model number: "+ str(m))
        Candidatos=[]       
        inicio=time.time()       
        error, epoca, modelo,errorAvgDays, FileVerification=modelConstruct_R(n_repeats,lista,input_num,fachl,
                             stop_tolerance,parametros_de_entrada,epocas,
                             nfolds,path,path_plot, train_vect,verbose,
                             var_order,path_file_obs,
                             path_file_appr,resol,kep_period,var_set,
                             input_unit,inputI,output_num,sets_unit,
                             train_set,valid_set,test_set,var_model,
                             var_angular,rem_trend,vect_step,
                             ResumenEjecucion,t_dist,m,
                             PathEIGEN,Functions,path_cpp)
                
                
    ##########################################################################################################
    ###### Saves the created model information and data in CandidatesSave.
    ##########################################################################################################

        Candidatos.append([Nombre[:-3],lista,error, errorAvgDays, epoca, round(time.time()-inicio,2)])
 
        modelo.save(path_path_models+Nombre)
        CandidatosA=pd.DataFrame(Candidatos)
        CandidatosA.columns=ColumnsCandidatos
        CandidatosSave=CandidatosSave.append(CandidatosA, ignore_index=True)
        del CandidatosA
        CandidatosSave.to_csv(fileCandidato)
        ######################################
        origen  = path+path_cpp+FileVerification
        if "exe" in FileVerification:
            destino = path_path_models+NombreCMasMas+".exe"
        else:
            destino = path_path_models+NombreCMasMas

        if os.path.exists(origen):
            with open(origen, 'rb') as forigen:
                with open(destino, 'wb') as fdestino:
                    shutil.copyfileobj(forigen, fdestino)
                    print("File with name "+ origen+" has been copied")    
        os.remove(origen) 

    print("Model: ",m,'\n')
    gc.get_threshold()
    G = gc.collect()
    print('Garbage collector has collected %d objects ' % (G)+'\n')
    return CandidatosSave


def roundBy(a, b):
  return int(round((a%b)/b)*b)+(a - (a%b))

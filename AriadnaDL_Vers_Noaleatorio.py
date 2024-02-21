# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:32:17 2021

# Versión completa I

# ARIADNA HYBRID PROPAGATION
# MODELING & FORECASTING
@author: edsegualv && hacarrh
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# COMMENTS
# ===================================================================

# TIME UNITS:
#  -Ephemeris files ("t"):            [day]
#  -Position error instants:          [day]
#  -Train, validation and test spans: [h]
#  -Keplerian period:                 [h]
#  -Resolutions:                      [min]

# ###################################################################

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Required libraries
# ===================================================================
import os, time, sys, gc, random, json, re, shutil
from distutils.util import strtobool
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import subprocess
import platform
#Validar requisitos del script
from sinfo import sinfo

Plataforma=platform.platform()

now = datetime.now()
print(now)

#######
# (1) #
#######
try:
    with open('DatosEntradaAriadnaDL_Noaleatorio.txt', 'r') as jsonfile:
        jsondata = ''.join(line for line in jsonfile if not line.startswith('#'))
        DatosEntradaRedIndividual = json.loads(jsondata)
except:
    exit("The input values of the experiment could not be read correctly.")

Modeling            = DatosEntradaRedIndividual["Modeling"]
Forecast            = DatosEntradaRedIndividual["Forecast"]
path_data           = DatosEntradaRedIndividual["path_data"]
plot_width          = DatosEntradaRedIndividual["plot_width"]
plot_height		    = DatosEntradaRedIndividual['plot_height']
t_dist		        = DatosEntradaRedIndividual['t_dist']
experimento         = DatosEntradaRedIndividual["experiment"]
var_set		        = DatosEntradaRedIndividual['var_set']
var_order		    = DatosEntradaRedIndividual['var_order']
var_model		    = DatosEntradaRedIndividual['var_model']
var_lab_unit		= DatosEntradaRedIndividual['var_lab_unit']
input_unit		    = DatosEntradaRedIndividual['input_unit']
inputI		        = DatosEntradaRedIndividual['inputI']
output_num		    = DatosEntradaRedIndividual['output_num']
sets_unit		    = DatosEntradaRedIndividual['sets_unit']
train_set		    = DatosEntradaRedIndividual['train_set']
valid_set		    = DatosEntradaRedIndividual['valid_set']
test_set	    	= DatosEntradaRedIndividual['test_set']
resol		        = DatosEntradaRedIndividual['resol']
kep_period		    = DatosEntradaRedIndividual['kep_period']
vect_step		    = DatosEntradaRedIndividual['vect_step']
nfolds		        = DatosEntradaRedIndividual['nfolds']
Parametros  		= DatosEntradaRedIndividual['ParametrosDL']
CONFIG  		    = DatosEntradaRedIndividual['CONFIGDL']
verbose		        = DatosEntradaRedIndividual['verbose']
fachl		        = DatosEntradaRedIndividual['fachl']
epocas		        = DatosEntradaRedIndividual['epocas']
n_repeats		    = DatosEntradaRedIndividual['n_repeats']
method		        = DatosEntradaRedIndividual['method']
NModels		        = DatosEntradaRedIndividual['NModels']
SampleMethod        = DatosEntradaRedIndividual['SampleMethod']
VariableToStratify  = DatosEntradaRedIndividual['VariableToStratify']
stop_tolerance		= DatosEntradaRedIndividual['stop_tolerance']
BasesAModelar		= DatosEntradaRedIndividual['BasesAModelar']
Intervalo		    = DatosEntradaRedIndividual['Intervalo']
fore_set_pos		= DatosEntradaRedIndividual['fore_set_pos']

var_angular         = strtobool(DatosEntradaRedIndividual["var_angular"])
rem_trend		    = strtobool(DatosEntradaRedIndividual['rem_trend'])

Functions           = DatosEntradaRedIndividual["Functions"]
PathEIGEN           = DatosEntradaRedIndividual["PathEIGEN"]+'"'
CodigosPythonPath   = DatosEntradaRedIndividual['CodigosPythonPath']
sys.path.append(CodigosPythonPath)


try:
  os.stat(CodigosPythonPath+'CodigosPython/')
  print("CodigosPython existe")
except:
  os.mkdir(CodigosPythonPath+'CodigosPython/')
  print("CodigosPython ha sido creada")

origen  = os.getcwd() + '/FuncionesAriadnaDL_Noaleatorio.py'
destino = CodigosPythonPath + 'CodigosPython/FuncionesAriadnaDL_Noaleatorio.py'

if os.path.exists(origen):
    with open(origen, 'rb') as forigen:
        with open(destino, 'wb') as fdestino:
            shutil.copyfileobj(forigen, fdestino)
            print("Archivo copiado")
import CodigosPython.FuncionesAriadnaDL_Noaleatorio as FAVCI



fore_set_posibilities=[                     
    ["train"],
    ["valid"],      
    ["test"],    
    ["train-valid"],
    ["valid-test"],
    ["train-valid-test"],
    ["train","test"],
    ["train","valid"],
    ["train","valid","test"],
    ["train","test","train-valid-test"],     
    ["train","test","valid-test","train-valid-test"],
    ["train-test"],
    ["train","train-test"],
    ["train","valid-test","test"]
    ]
fore_set = fore_set_posibilities[fore_set_pos]


sinfo()
#######
# (2) #
#######


if Intervalo==1:
    BasesAModelar=range(BasesAModelar[0],BasesAModelar[1]+1)  
else:
    BasesAModelar=BasesAModelar  

###################################################################################################################
###### START TRAINING AND FORECASTING
###################################################################################################################

I_CreaInfo=time.time()

for h1 in BasesAModelar:
    
    print(var_set+"-"+str(h1)+" Time series modelling.")    
    ##########################################################################################################
    ###### NAME OF EPHEMERID TO BE STUDIED
    ##########################################################################################################
    id_expM = var_set+str(h1)    
    ##########################################################################################################
    ###### CREATE NAME OF DIRECTORY WITH PARAMETERS
    ##########################################################################################################
    experiment=str(experimento)+"_"+"Fachl"+str(fachl) +"Me"+str(method)+"TrS"+str(train_set) + "VaS"+str(valid_set)+"TeS"+str(test_set)+"Res"+str(resol)+"Nf"+str(nfolds)+"Rt"+str(rem_trend) +"VarS"+str(var_set)+"Ep"+str(epocas)+"St"+str(stop_tolerance)+"VarM"+str(var_model)
    ###########################################################################################################
    ###### Create folder name
    ###########################################################################################################
    Conid_exp = id_expM + experiment        
    cwd=os.getcwd()
    cwd2=cwd
    ##########################################################################################################
    ###### Configuration of files and folders
    ##########################################################################################################
    try:
      os.stat('KerasPruebasFinal/')
    except:
      os.mkdir('KerasPruebasFinal/')
    
    path= 'KerasPruebasFinal/'+Conid_exp+'/'
    
    path_results           = "results/"
    path_out               = "output/"
    ResumenEjecucion       = path+"output/Log.txt"
    path_plot              = "figs/"
    path_models            = "Modelos/"
    path_cpp               = "cpp/"
    file_obs               = 'obs'+id_expM+'.out'
    file_appr              = 'approx'+id_expM+'.out'
    file_BasicPlot         = id_expM +"_BasicPlot"
    file_plot_Decomp       = id_expM +"_Decomp"
    file_plot_Complete     = id_expM +"_Completo"
    fileCandidato          = path+path_results+'Candidatos.csv'
    path_path_models       = path+path_models
    path_file_obs          = path_data+"OBS/"+file_obs
    path_file_appr         = path_data+"APPROX/"+file_appr
    path_file_BasicPlot    = path+path_plot + file_BasicPlot +'.pdf'
    fileplotDetrending     = path+path_plot + file_plot_Decomp+".pdf"
    fileplotComplete       = path+path_plot + file_plot_Complete+".pdf"
    directorio             = ['figs','Modelos','output','results', "cpp"]
    
    #Grid Comfigurations:
    hp_all=FAVCI.DeepLerningModelConfigs(Parametros)
    NModTotal=len(hp_all)
    random.seed(2022)
    hp_all=random.sample(hp_all, len(hp_all))
        
    if method==1:
        NModels=NModTotal
        hp_all=hp_all
        Metodo="Cartesiana"        
    elif method==2:   
        Metodo="Aleatoria"
        print("Hiper-parameter to sampling: "+ VariableToStratify)       
        ListaDeOpciones=Parametros[VariableToStratify]  
        hp_all=[y for x in ListaDeOpciones for y in hp_all if x==y[VariableToStratify]]        
        StrVar=[str(hp_all).count("'"+VariableToStratify+"': '"+str(n)+"'") if type(n)==str else str(hp_all).count("'"+VariableToStratify+"': "+str(n)) for n in ListaDeOpciones]
                
        if SampleMethod==0: 
            if NModels< len(StrVar):
                NModels=len(StrVar)
            else:
                NModels=FAVCI.roundBy(NModels, len(StrVar))           
            indices=[]
            nPerHipePar=round(NModels/len(StrVar))
            a=0
            for i in StrVar:
                random.seed(2022)
                indices=indices+random.sample(range(a, i+a), nPerHipePar)
                a=i+a       
            hp_all=[hp_all[index] for index in indices]
        else:
            indices=[]
            nPerHipePar=[round(NModels*(x/sum(StrVar))) for x in StrVar]
            a=0
            zipped = list(zip(StrVar,nPerHipePar))
            for i in zipped:
                print(i)
                random.seed(2022)
                X=random.sample(range(a, i[0]+a), i[1])
                indices=indices+X                
                a=i[0]+a                 
            # indices.sort()
            hp_all=[hp_all[index] for index in indices]
            
            
    elif method==3:
        Metodo="Modelo Específico"
        # hp_all = [CONFIG]
        hp_all = CONFIG
        NModels=len(hp_all)
        
    print("Total number of possible models: "+ str(NModTotal)+"\n"+
          "Number of models to find: "+ str(len(hp_all)))
    ###########################################################################################################
    ###### FOLDER CREATION AND VALIDATION
    ###########################################################################################################
    # Check if a folder with the same name as the current experiment already exists. 
    # If not, a folder with that name is created.                                                          
    ###########################################################################################################
    
    Verif=path_results
    
    if os.path.exists(path):       
            
    ###############################################################
    # It is verified that the existing folder is of a complete or 
    # finished experiment, otherwise it is deleted.
    # experiment, otherwise the folder is deleted.   
    ###############################################################
       
        if os.path.exists(path+Verif) and len(path_path_models)==NModTotal:
                
            sub=1
            repeticion=""
            NuevoArchivo="False"
                
                
            while(NuevoArchivo=="False"):
                    
    #############################################################################
    # If the existing folder has a completed experiment 
    # then folders are searched for folders with the same experiment but
    # performed later. What is identified by (i), if found, is 
    # verify that they are complete. Otherwise, they are deleted and the current 
    # is assigned that name. "os.path.exists(path)" is checked 
    # again since in each iteration of while, it changes due to "sub". 
    #############################################################################
     
                if  os.path.exists(path):
                        
                    if len(os.listdir(path+Verif))>0:
                        repeticion="(" +str(sub)+")" +'/' 
                        path= 'KerasPruebasFinal/'+Conid_exp+repeticion
                        sub+=1
                        
                    else:
                        
                        print(f"Borrando directorio vacio 1 {path}."+'\n')
                        shutil.rmtree(path)
                        os.mkdir(path)
                        os.chdir(path)
                        
                
                        for dir in directorio:
                            try:
                                os.stat(dir)
                            except: 
                                os.mkdir(dir)
                                
                        NuevoArchivo="True"
                        os.chdir(cwd2) 
                            
                else:
                    
                    os.mkdir(path)
                    os.chdir(path)
                    
            
                    for dir in directorio:
                            try:
                                os.stat(dir)
                            except: 
                                os.mkdir(dir)
                    NuevoArchivo="True"
                    os.chdir(cwd2) 
                        
              
                    
        elif os.path.exists(path+Verif) and len(path_path_models)<NModTotal:    
            pass
        else:
            
            print(f"Borrando directorio vacio 2 {path}."+'\n')
            shutil.rmtree(path)
            os.mkdir(path)
            os.chdir(path)
            
            
            for dir in directorio:
                
                try:
                    os.stat(dir)
                except:
                    os.mkdir(dir)
        
            os.chdir(cwd2)        
    else: 
        
        os.mkdir(path)
        os.chdir('KerasPruebasFinal/'+Conid_exp)
                   
        for dir in directorio:
            
            try:
                os.stat(dir)
            except:
                os.mkdir(dir)
                
        os.chdir(cwd2)
   
    
    if not os.path.isfile(path+path_out+"hp_all.csv"):
        pd.DataFrame(hp_all).to_csv(path+path_out+"hp_all.csv")
    #########################################################################      
    #                End of Folder Verification                             #       
    #########################################################################    
    if not os.path.isfile(ResumenEjecucion):
        Start=str(
          "#############################################################################\n"+
          "Start modeling: \n"+
          "#############################################################################\n"+
          "Start time: "+ str(datetime.now()) +"\n"+
          "Folder Name: "+ str(Conid_exp) +"\n"+
          "Grid Search Method: " + str(Metodo) + "\n"+
          "Number of models: "+ str(NModels) +"\n"+
          "Maximum number of epochs: "+str(epocas)+ "\n"+
          "Stop tolerance: "+ str(stop_tolerance) +"\n"+
          "#############################################################################\n"+
          "#############################################################################\n"      
          )  
        with open(ResumenEjecucion, 'a') as f:
            print(Start
          , file=f)  
        print(Start)
    
    ##########################################################################################################
    ###### CREATING DATABASES
    ##########################################################################################################
    I_bases=time.time()
    Bases=FAVCI.creaBases(var_order,path_file_obs,path_file_appr,resol,kep_period,var_set,input_unit,inputI,
                          output_num,nfolds,sets_unit,train_set,test_set,var_model,rem_trend,vect_step,
                          ResumenEjecucion,valid_set)
    
    
    
    ind_train=Bases[0]
    train_seq=Bases[1]
    train_seq_num=Bases[2]
    train_trend_seq=Bases[3]
    train_vect=Bases[4]
    ind_valid=Bases[5]
    ind_valid0=Bases[6]
    valid_seq=Bases[7]
    valid_seq_num=Bases[8]
    valid_trend_seq=Bases[9]
    valid_vect=Bases[10]
    ind_test=Bases[11]
    ind_test0=Bases[12]
    test_seq=Bases[13]
    test_trend_seq=Bases[15]
    appr=Bases[16]
    obs=Bases[17]
    error=Bases[18]
    error_comps=Bases[19]
    error_trend=Bases[20]
    errorBU=Bases[21]
    input_num=Bases[22]
    
    print('Time creating data '+str(int(time.time()-I_bases))+' sec'+'\n')
    
    ##########################################################################################################
    ###### MAIN GRAPHICS
    ##########################################################################################################
    g_time=time.time()
    
    if not os.path.isfile(path_file_BasicPlot):
        FAVCI.BasicPlot(var_model,path_file_BasicPlot,plot_width,plot_height,train_seq,input_num,
                        output_num,train_seq_num,test_seq,nfolds,valid_seq,var_lab_unit)
    
    if (rem_trend):
    
        train_seq_t  =  pd.concat([train_seq['t'], train_seq[var_model]-train_trend_seq[var_model]], axis=1)
        valid_seq_t  =  pd.concat([valid_seq['t'], valid_seq[var_model]-valid_trend_seq[var_model]], axis=1)
        test_seq_t   =  pd.concat([test_seq['t'], test_seq[var_model]-test_trend_seq[var_model]], axis=1)
    
        if not os.path.isfile(fileplotDetrending):
            FAVCI.plot_decomp(fileplotDetrending,error_comps)
        if not os.path.isfile(fileplotComplete):
            FAVCI.plot_complete(fileplotComplete,errorBU,var_model,error_trend,error,var_lab_unit)
            
        FAVCI.BasicPlot(var_model,path_file_BasicPlot,plot_width,plot_height,train_seq_t,input_num,
                        output_num,train_seq_num,test_seq_t,nfolds,valid_seq_t,var_lab_unit)    
    
    print('Time main graphics '+str(int(time.time()-g_time))+' sec'+'\n')
    
    ##########################################################################################################
    ###### Call KERAS.SERIE function (Tensorflow + keras)
    ##########################################################################################################
   
    collected = gc.collect() 
    print("Garbage collector start: collected %d objects." % (collected))

         
    ###########################################################################################################
    ###########################################################################################################     
    #### Modelling ####
    I_Modelado=time.time()
    if Modeling==1:
        for lista in hp_all:
            
            Candidatos= FAVCI.KerasSerie(train_vect,valid_vect,fileCandidato,path_path_models,
                           n_repeats,lista,input_num,fachl,stop_tolerance,epocas,
                          nfolds,path,path_plot,verbose,var_order,path_file_obs,
                          path_file_appr,resol,kep_period,var_set,input_unit,inputI,
                          output_num,sets_unit,train_set,valid_set,test_set,var_model,
                          var_angular,rem_trend,vect_step,ResumenEjecucion,t_dist,
                          PathEIGEN,Functions,path_cpp) 
            print("Length Candidates: "+str(Candidatos.shape[0]))
            collected = gc.collect() 
            print("Garbage collector end: collected %d objects." % (collected))
            
            gc.get_count()
            gc.collect()        
            gc.get_count()    
        
        print("Modeling time: ", str(round(time.time()-I_Modelado,2))+'\n')   
    else:
        
        ModStr=str(
          "#############################################################################\n"+
          "Candidatos file has been imported (Not modelling)"+
          "#############################################################################\n"+
          "Start time again: "+ str(datetime.now()) +"\n"+
          "Folder Name: "+ str(Conid_exp) +"\n"+
          "#############################################################################\n"+
          "#############################################################################\n"      
          )         
            
        with open(ResumenEjecucion) as f:
            if not 'Candidatos file has been imported' in f.read():
                with open(ResumenEjecucion, 'a') as f:
                    print(ModStr, file=f)  
        print(ModStr)
        if os.path.isfile(fileCandidato):
            Candidatos=pd.read_csv(fileCandidato, index_col=0)
###########################################################################################################
###########################################################################################################    
#### Forecasting ####

    I_Forecasting=time.time()    
    filenamedistKMHyb=path+path_results+"DistKMhyb.xlsx"    
    # RMSE = pd.DataFrame(columns=["FileName","Rmse_Epsilon","Hill", "fore_set"])


    if not os.path.isfile(filenamedistKMHyb):
        DistKMhybF = pd.DataFrame(columns=["Valida"])
    else:
        DistKMhybF = pd.read_excel(filenamedistKMHyb,engine='openpyxl', index_col=0)

#Create file "LeerPesosdetxt.h"   

    if not os.path.isfile(path_path_models+"LeerPesosdetxt.h"):
        FAVCI.LeerPesosdetxt("",path_path_models)  
    
    for m in Candidatos.index:
        mini=time.time()
        lista=Candidatos.iloc[m,Candidatos.columns == 'Candidato'].values[0]
        FileName = "modelo"+str(m)
        Mod=tf.keras.models.load_model(path_path_models+FileName+".h5", compile=False)
        
        #Extract Model information
        NCapas,Act,K=FAVCI.ExtractModInfo(Mod,ResumenEjecucion)
        
        for fores in fore_set:
            filenamedistKMAppr=path+path_results+"DistKMAppr"+fores+".xlsx"
            filenamedistKMApprBest=path+path_results+"DistKMApprBest"+fores+".xlsx"  
            Valida=str(lista)+"_"+fores+"_m_"+str(m)
            if not Valida in list(DistKMhybF["Valida"]):                
                ini=time.time()
                fore_seq,fore_test,obs_test,appr_test,fore_trend_seq,Reales,Reales_trend=FAVCI.Select_fore_set(fores,
                ind_train,nfolds,ind_valid0,train_seq_num,input_num,output_num,
                ind_test0,valid_seq_num,error,errorBU,error_trend,ind_valid,ind_test,obs,appr,var_model)       
            
                #Export input files for C++ execution
                FAVCI.ExportDataInputCMasMas("",path_path_models,fore_seq,input_num,K,Mod)       
    ########################################################################################################### 
                Epsilon=Reales.loc[input_num:,var_model]
                Epsilon.to_csv(path+path_results+"Epsilon"+fores+".csv")
                
                #creando modelo c++
                #Crea archivo LeerPesosdetxt.h  
                Inicio=time.time()
                if not os.path.isfile(path_path_models+"LeerPesosdetxt.h"):
                    FAVCI.LeerPesosdetxt("",path_path_models)               
                           
 
                ######################################
                #### Crea archivo RedNeuronal.cpp
                if not os.path.isfile(path_path_models+FileName+".cpp"):  
                    FAVCI.creaRedNeuronalcMasMas(path_path_models+FileName+".cpp",K,Act,NCapas,Functions)
                ######################################
                    
                ######################################
                #### Compile RedNeuronal.cpp file 
                Inicio=time.time()
                Plataforma=platform.platform()
                if Plataforma.find("indows") == 1:
                    FileVerification=FileName+".exe"
                else:
                    FileVerification=FileName
                if not os.path.isfile(path+path_cpp+FileVerification):
                    cwd=os.getcwd()
                    cwd2=cwd                
                    os.chdir(os.path.abspath(path_path_models))
                    
                    #O1, O2 or O3 can be used. As it increases it loses accuracy. 
                    if Plataforma.find("indows") == 1:
                        pl=subprocess.Popen(r"g++ -O2 -I EINGEN/eigen/ "+FileName+ ".cpp -o "+FileName,shell=True)  
                    else:
                        pl=subprocess.Popen(r"g++ -std=c++0x -O2 -I EINGEN/eigen/ "+FileName+ ".cpp -o "+FileName,shell=True)  
                    pl.wait()
                    os.chdir(cwd2)
                ##############################################
    
                fore_timeI=time.time()  
                DistKMAppr,DistKMApprBest,DistKMhyb,Fore,Cols=FAVCI.forec(t_dist,fore_seq,obs_test,appr_test, Mod,fore_test,m,input_num,
                          output_num,var_model,var_angular,var_set,FileName,Functions,"",path_path_models)
                fin=time.time()
                
                #RMSE:        
                
                Epsilon_Fore=Fore.loc[input_num:,var_model]
                Epsilon_Fore.to_csv(path+path_results+"Epsilon_Fore"+fores+str(m)+".csv")
                Rmse_Epsilon=FAVCI.RMSE(Epsilon, Epsilon_Fore)     
              
                
                print("Modeling + forecasting time: ", str(round(fin-ini,2))+'\n')
    ###########################################################################################################
                DistKMhyb[DistKMhyb == 0] = np.nan
                DistKMhyb["Avg_Days_Km"]=DistKMhyb[Cols].mean(axis=1,skipna = True)            
                DistKMhyb["fore_set"]=fores
                DistKMhyb["Valida"]=Valida
                DistKMhyb["Valida"]=Valida
                DistKMhyb["Time"]=round(time.time()-fore_timeI,2)
                DistKMhyb["Rmse_Epsilon"]=Rmse_Epsilon
                
                
                if not os.path.isfile(filenamedistKMAppr):
                    DistKMAppr=DistKMAppr.to_excel(filenamedistKMAppr, index = True)      
                if not os.path.isfile(filenamedistKMApprBest):
                    DistKMApprBest=DistKMApprBest.to_excel(filenamedistKMApprBest, index = True)     
                
                if (rem_trend):
                    Fore[var_model]  = Fore[var_model] + fore_trend_seq[var_model]
                    Reales           = Reales_trend
                
                DistKMhybF=DistKMhybF.append(DistKMhyb)                
                DistKMhybF.to_excel(filenamedistKMHyb, index = True)
                
                nombregrafico=path+path_plot+id_expM +"_"+fores+"_m_"+str(m)
                if not os.path.isfile(nombregrafico):   
                    FAVCI.plot_fore(Fore,Reales,nombregrafico,plot_width,plot_height,input_num,output_num,var_model,var_lab_unit)
                
                if os.path.isfile(path_path_models+"Entrada.csv"):        
                    os.remove(path_path_models+"Entrada.csv")  
                if os.path.isfile(path_path_models+'NumeroDatos.txt'):      
                    os.remove(path_path_models+'NumeroDatos.txt')   
                if os.path.isfile(path_path_models+"Fore.dat"):  
                    os.remove(path_path_models+"Fore.dat")    
                
                collected = gc.collect() 
                print("Garbage collector FORECASTING: collected %d objects." % (collected))
                
                gc.get_count()
                gc.collect()        
                gc.get_count()    
                  
        #Del c++ inputs files               
        for k in K:
            a_file = path_path_models+k+".csv"
            if os.path.isfile(a_file):
                os.remove(a_file)
        gc.get_count()
        gc.collect()        
        gc.get_count()
    
    print("Fore time: ", str(round(time.time()-I_Forecasting,2))+'\n')   
    
    #Eliminar carpeta Eigen
    if os.path.exists(path_path_models+"EINGEN"):
        try:
            shutil.rmtree(path_path_models+"EINGEN")
            print('EINGEN directory deleted')
        except OSError as err:
            print("Except: \n")
            print(err)
   


################################################################################################
################################################################################################              
### SUMMARY:
################################################################################################
################################################################################################                

    if not method==3 and Forecast==1:
        with open(ResumenEjecucion) as f:
            if 'BEST MODEL' in f.read():
                # Delete some text in a text file
                orig_file = open(ResumenEjecucion, "r")
                lines = orig_file.readlines()
                 
                start = False
                saved_list = []
                for rec in lines:
                    if "BEST MODEL:" in rec:
                        start = True
                    if not start:
                        saved_list.append(rec)
                    if "BEST MODEL END" in rec:
                        start= False
                        
                new_text_file = open(ResumenEjecucion, "w")
                new_text_file.writelines(saved_list)
                new_text_file.close()
                
        Mejor=pd.DataFrame(Candidatos.sort_values(by=['errorAvgDays'])).head(1)
        try:
            Resumen=str(
              "#############################################################################\n"+
              "BEST MODEL: \n"+
              "#############################################################################\n"+
              "The total time spent performing the gird search "+ Metodo +"of "+str(Candidatos.shape[0])+" models is: "+ str(sum([Candidatos["time"][a] for a in Candidatos.index if type(Candidatos["time"][a])!=str]))+ " seconds \n"+
              "The total forecast time is: "+ str(sum(DistKMhybF["Time"]))+ " seconds \n"+
              "#############################################################################\n"+
              "The best candidate is: \n" + str(Mejor["Candidato"].values[0]) +"\n"+
              "Such as: \n"+
              "Error" + Mejor["Candidato"].values[0]["loss"] + " is: "+ str(Mejor["Error"].values[0])+"\n"+
              "Average error in distance is: "+str(Mejor["errorAvgDays"].values[0])+"\n"
              "Number of epochs is : "+str(Mejor["Epochs"].values[0])+"\n"+
              "Model name is "+str(Mejor["Nombre"].values[0])+"\n"+        
              "End time: "+ str(datetime.now()) +"\n"+
              "#############################################################################\n"+
              "#############################################################################\n"+
              "#############################################################################\n"+
              "BEST MODEL END \n")  
        except:
            Resumen=str(
              "#############################################################################\n"+
              "BEST MODEL: \n"+
              "#############################################################################\n"+
              "The total time spent performing the grid search  "+ Metodo +"of "+str(Candidatos.shape[0])+" models is: "+ str(sum([Candidatos["time"][a] for a in Candidatos.index if type(Candidatos["time"][a])!=str]))+ " seconds \n"+
              "The total forecast time is: "+ str(sum(DistKMhybF["Time"]))+ " seconds \n"+
              "#############################################################################\n"+
              "The best candidate is: \n" + str(Mejor["Candidato"].values[0]) +"\n"+
              "Such as: \n"+
              "Error" + re.search('%s(.*)%s' % ("'loss':", ", 'LR'"), Mejor["Candidato"].values[0]).group(1) + " is: "+ str(Mejor["Error"].values[0])+"\n"+
              "Average error in distance is: "+str(Mejor["errorAvgDays"].values[0])+"\n"
              "Number of epochs is : "+str(Mejor["Epochs"].values[0])+"\n"+
              "Model name is "+str(Mejor["Nombre"].values[0])+"\n"+              
              "End time: "+ str(datetime.now()) +"\n"+
              "#############################################################################\n"+
              "#############################################################################\n"+
              "#############################################################################\n"+
              "BEST MODEL END \n")  
        with open(ResumenEjecucion, 'a') as f:
            print(Resumen
          , file=f)  
        print(Resumen) 
        
print("Hora inicio:")
print(now)
print("Hora fin:")
print(datetime.now())
        

#Eliminar carpeta Eigen
if os.path.exists(path+path_cpp+"EINGEN"):
    try:
        shutil.rmtree(path+path_cpp+"EINGEN")
        print('EINGEN directory deleted')
    except OSError as err:
        print("Except: \n")
        print(err)
        
        

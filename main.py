#Correct python interpreter: usr/local/bin
from statistics import LinearRegression
import matplotlib
import seaborn as sns
import csv
import re
import statsmodels
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json

#atomic mass, entropy, electron affinity
dt1 = pd.read_csv('train.csv')
#critical temperature, (charge carrier density)
dt2 = pd.read_csv('T_c&P&Q&Debye.csv')
#critical temperature, doping, pressure
dt3 = pd.read_csv('SuperCon.csv')
#critical temperature, electron-phonon coupling parameter
dt4 = json.load(open('jarvis_phonon.json'))

#DOS in download7.csv

T_c = dt3['T_C'] #find the best and largest resource (largedata.csv)　　　　　　　 　　 
T_atm = dt1['critical_temp']                                                         
atomass = dt1['mean_atomic_mass']                                                    
entropy = dt1['wtd_entropy_atomic_mass']                                             
radius = dt1['wtd_mean_atomic_radius']                                               
affinity = dt1['mean_ElectronAffinity']                                              
pressure = dt3['appliedPressure'] #pressure in VARIOUS UNITS　　　　　　　　　　　　　　　
#put coupling into a list                     #take magnitudes of vector lists, match using rounded T_c

#get DOS data (worry about later)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

#use gaussian approximation to fill in P values for secondary enhanced analysis (but add disclaimer)


# Note: When intersecting/mapping datasets, DO NOT USE PRESSURE-APPLIED T_C <==> NATURAL T_C
# Use planned algorithm to match & intersect [training] with [SuperCon]

#match datasets using mean atomic mass
atomic_masses = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.811,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.086,
    "P": 30.974,
    "S": 32.065,
    "Cl": 35.453,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.380,
    "Ga": 69.723,
    "Ge": 72.640,
    "As": 74.922,
    "Se": 78.960,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.620,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.960,
    "Tc": 98.000,
    "Ru": 101.070,
    "Rh": 102.906,
    "Pd": 106.420,
    "Ag": 107.868,
    "Cd": 112.411,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.600,
    "I": 126.904,
    "Xe": 131.293,
    "Cs": 132.905,
    "Ba": 137.327,
    "La": 138.905,
    "Ce": 140.116,
    "Pr": 140.908,
    "Nd": 144.242,
    "Pm": 145.000,
    "Sm": 150.360,
    "Eu": 151.964,
    "Gd": 157.250,
    "Tb": 158.925,
    "Dy": 162.500,
    "Ho": 164.930,
    "Er": 167.259,
    "Tm": 168.934,
    "Yb": 173.054,
    "Lu": 174.967,
    "Hf": 178.490,
    "Ta": 180.948,
    "W": 183.840,
    "Re": 186.207,
    "Os": 190.230,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.967,
    "Hg": 200.590,
    "Tl": 204.383,
    "Pb": 207.200,
    "Bi": 208.980,
    "Po": 209.000,
    "At": 210.000,
    "Rn": 222.000,
    "Fr": 223.000,
    "Ra": 226.000,
    "Ac": 227.000,
    "Th": 232.038,
    "Pa": 231.036,
    "U": 238.029,
    "Np": 237.000,
    "Pu": 244.000,
    "Am": 243.000,
    "Cm": 247.000,
    "Bk": 247.000,
    "Cf": 251.000,
    "Es": 252.000,
    "Fm": 257.000,
    "Md": 258.000,
    "No": 259.000,
    "Lr": 262.000,
    "Rf": 267.000,
    "Db": 268.000,
    "Sg": 269.000,
    "Bh": 270.000,
    "Hs": 269.000,
    "Mt": 278.000,
    "Ds": 281.000,
    "Rg": 282.000,
    "Cn": 285.000,
    "Nh": 286.000,
    "Fl": 289.000,
    "Mc": 290.000,
    "Lv": 293.000,
    "Ts": 294.000,
    "Og": 294.000}
def mean_molecular_mass(formula):
    pattern = r'([A-Z][a-z]?)\s*([0-9]*\.?[0-9]*)'
    matches = re.findall(pattern, formula)
    total_mass = 0.0
    num_atoms = 0
    
    for element, quantity in matches:
        if element in atomic_masses.keys():
            if quantity == '':
                quantity = '1'
            quantity = float(quantity)
            element_mass = atomic_masses[element] * quantity
            total_mass += element_mass
            num_atoms += quantity
    
    if num_atoms==0: return -1
    return round(total_mass/num_atoms,1)

P = [] #pressure converted to Pa

def convert(unit):
    match unit:
        case ' mtorr':
            return 133322368
        case ' gpa':
            return 1000000000
        case 'gpa':
            return 1000000000
        case ' pa':
            return 1
        case 'pa':
            return 1
        case 'mbar':
            return 100000000000
        case ' mbar':
            return 100000000000
        case _:
            return 1000000000
        
T_actual = []
indices1 = []
dt3masses = []

for i in range(18940):
    x_step = re.findall(r'\d+', str(pressure[i]))
    if (not str(pressure[i]).isalpha()) and x_step:
        y_step = float(x_step[0])
        z_step  = y_step * convert(str(dt3['appliedPressureUnit'][i]).lower())
        if z_step>1.01325e+05: #weeding out invalid/redundant values
            P.append(z_step)
            T_actual.append(float(T_c[i]))
            dt3masses.append(mean_molecular_mass(dt3['formula'][i]))
            indices1.append(i)

mapping = {}
mapping2 = {}
couplings = []

for n in range(941):
    print('mapping '+str(n))
    for j in range(21263):  
        if dt3masses[n]==round(atomass[j],1): #make it so that atomass[j] fetches dt3masses[n]
            mapping[n] = j #n is the index in T_actual, i in dt3, and j in atomass

for n in range(1063):
    print('mapping '+str(n))
    for j in range(21263):
        coupler = mean_molecular_mass(''.join(dt4[n]["atoms"]["elements"]))
        if j==0: couplings.append(coupler) #adding into array for first time

        if coupler==round(atomass[j],1):
            mapping2[n] = j

A_actual = []
E_actual = []
R_actual = []
EA_actual = []
T_atm_actual = []
#EPhC_actual = []

total_len = 0
matchez = []
for i in range(940):
    print("trying "+str(i))
    if (i in mapping.keys()):
        total_len+=1
        matchez.append(i)

        w_step = atomass[mapping[i]]
        v_step = entropy[mapping[i]]
        u_step = 0.01*radius[mapping[i]]
        t_step = re.findall(r'\d+', str(T_atm[mapping[i]]))
        s_step = affinity[mapping[i]]
        #r_step = couplings[mapping2[i]]

        A_actual.append(w_step)
        E_actual.append(v_step)
        R_actual.append(u_step)
        EA_actual.append(s_step)
        #EPhC_actual.append(r_step)
        T_atm_actual.append(t_step[0])


print("TOTAL LENGTH: "+str(total_len))

#creating the new CSV
rows = []
for i in range(total_len):
    rows.append({'T_exp': T_actual[matchez[i]], 'P': P[matchez[i]], 'AM': A_actual[i], 'S': E_actual[i], 'R': R_actual[i], 'EA': EA_actual[i], 'T_atm': T_atm_actual[i]})   #'Phonon Coupling': EPhC_actual[i]

with open('superconductor8.csv', 'w', newline='') as csvfile:
    variables = ['T_exp', 'P', 'AM', 'S', 'R', 'EA', 'T_atm']
    writer = csv.DictWriter(csvfile, fieldnames=variables)
    writer.writeheader()
    writer.writerows(rows)

#INTERESTING CASE: 61ba9ebe85b12ae79ac512a4,H 3 S,-386152408,,H3S,,,Hydrides. Chalcogenides,,,,200.0,K,,160200000000000.0,Pa,body,paragraph,42d5826002,Mechanism of High-Temperature Superconductivity in Correlated-Electron Systems,10.3390/condmat4020057,Takashi Yanagisawa,MDPI AG,Condensed Matter,2019

# ---Cuprates Plotter---
# X = np.array(P).reshape(-1,1)
# y = np.array(T_actual).reshape(-1,1)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=100)

# #applying polynomial regression
# poly = PolynomialFeatures(degree=5, include_bias=True)
# x_train_trans = poly.fit_transform(x_train)
# x_test_trans = poly.transform(x_test)
# #include bias parameter
# lr = LinearRegression()
# lr.fit(x_train_trans, y_train)
# y_pred = lr.predict(x_test_trans)

# print("Polynomial Coefficients: ")
# print(lr.coef_)
# print(lr.intercept_)

# #plotting
# print("Number of P values: "+str(len(P)))
# X_new = np.linspace(8, 12, 200).reshape(200, 1)
# X_new = 10**X_new
# X_new_poly = poly.transform(X_new)
# y_new = lr.predict(X_new_poly)
# plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
# plt.scatter(P, T_actual, s=10, color="blue",zorder=1)
# plt.xlabel("Pressure (Pa)")
# plt.ylabel("Critical Temperature (K)")
# plt.xscale('log')
# plt.yscale('log')
# #plt.axis([2e8, 1e12, 1e-4, 500])
# plt.legend()
# plt.show()
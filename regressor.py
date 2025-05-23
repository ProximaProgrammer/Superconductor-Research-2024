import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('superconductorV7.csv')
x1 = data['P']
x2 = data['AM']
x3 = data['S']
x4 = data['R']
x5 = data['EA']
x6 = data['T_atm'] #not included because accuracy is hardly improved
y = data['T_exp']
DEG_ACTUAL = 6

def polynomial_regression(xx1, xx2, xx3, xx4, xx5, y, DEG):
    # Stack features into matrix
    X_raw = np.column_stack((xx1, xx2, xx3, xx4, xx5))

    # Standardize inputs to maximize T_C prediction accuracy
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    poly = PolynomialFeatures(degree=DEG)
    X_poly = poly.fit_transform(X_scaled)
    reg = LinearRegression()
    reg.fit(X_poly, y)

    y_pred = reg.predict(X_poly)
    rrmse = np.sqrt(mean_squared_error(y, y_pred) / np.mean(y**2))
    print(DEG, "RRMSE:", rrmse)

    # plotting for a general idea
    x1_smooth = np.linspace(xx1.min(), xx1.max(), 200)
    X_smooth = np.column_stack([
        x1_smooth,
        [np.mean(xx2)] * 200,
        [np.mean(xx3)] * 200,
        [np.mean(xx4)] * 200,
        [np.mean(xx5)] * 200,
    ])
    X_smooth_scaled = scaler.transform(X_smooth)
    X_smooth_poly = poly.transform(X_smooth_scaled)
    y_smooth = reg.predict(X_smooth_poly)

    plt.scatter(xx1**(-2), y, alpha=0.5, label='Data Points')
    plt.plot(x1_smooth, np.exp(y_smooth), 'r--', label='Polynomial Fit')
    plt.xlabel('x1')
    plt.ylabel('T_exp')
    plt.title(f'Multivariate Polynomial Regression (RRMSE: {rrmse:.4f})')
    plt.legend()
    plt.grid(True)
    #plt.show()

    # Print coefficients
    coef = pd.DataFrame({
        'term': poly.get_feature_names_out(),
        'coefficient': reg.coef_
    })
    print("\nCoefficients:")
    print(coef)

    print("Model training complete.")

    rows = []
    for i in range(len(coef)):
        rows.append({'Term': poly.get_feature_names_out()[i], 'Coefficient': reg.coef_[i]})

    with open('coefficientsV9.csv', 'w', newline='') as csvfile:
        variables = ['Term', 'Coefficient']
        writer = csv.DictWriter(csvfile, fieldnames=variables)
        writer.writeheader()
        writer.writerows(rows)
    
    print("coefficients written.")
    return reg, poly, scaler

# Fit model
reg, poly, scaler = polynomial_regression(x1, x2, x3, x4, x5, y, DEG_ACTUAL)

'''
# Predict on new data as a test
X_input_raw = np.array([[155000000000.0,32.065,0.0,0.87,201.5,14]])
X_input_scaled = scaler.transform(X_input_raw)
X_poly_input = poly.transform(X_input_scaled)
y_pred = reg.predict(X_poly_input)
print("Predicted T_exp:", y_pred)


#T_c Estimation 

exp = data['T_exp']
atm = data['T_atm']
pressure = data['P']
C2 = []
C1 = []
counted = 0

for i in range(691):
    x = exp[i]
    p = pressure[i]
    a = atm[i]
    
    if (0<a<x):
        counted += 1

        #same parabolic shape, but shifted up/down depending on the superconductor (same horizontal dispersion)
        c = math.log((a**3.9)/x) # 4.855 = 3.9log(T_atm) - log(T_exp)
        #C2.append(c)
        C2.append((c-(4.854683252174617))**2)
        C1.append((c**2))

#RRMSE calculation
print(math.sqrt(sum(C1)/counted))
print(sum(C2)/sum(C1))

#Note: predict new superconductors from list of confirmed superconductors but with no recorded T_c, and also arbitrary materials separately
#Note: to keep everything the same pressure, use the gaussian approximation for pressure to enter in data for T_c. 
#          train.csv includes 1 atm T_c

'''
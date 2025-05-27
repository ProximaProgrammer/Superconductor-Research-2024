import numpy as np
import pandas as pd
import csv
import random
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error

# Load datasets
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
    #stack features into a matrix of data
    X_raw = np.column_stack((xx1, xx2, xx3, xx4, xx5))

    #standardize inputs to maximize T_c prediction accuracy
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

    # x1_smooth = np.linspace(xx1.min(), xx1.max(), 200)
    # X_smooth = np.column_stack([
    #     x1_smooth,
    #     [np.mean(xx2)] * 200,
    #     [np.mean(xx3)] * 200,
    #     [np.mean(xx4)] * 200,
    #     [np.mean(xx5)] * 200,
    # ])
    # X_smooth_scaled = scaler.transform(X_smooth)
    # X_smooth_poly = poly.transform(X_smooth_scaled)
    # y_smooth = reg.predict(X_smooth_poly)

    # plt.scatter(xx1**(-2), y, alpha=0.5, label='Data Points')
    # plt.plot(x1_smooth, np.exp(y_smooth), 'r--', label='Polynomial Fit')
    # plt.xlabel('x1')
    # plt.ylabel('T_exp')
    # plt.title(f'Multivariate Polynomial Regression (RRMSE: {rrmse:.4f})')
    # plt.legend()
    # plt.grid(True)
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
    
    print("Coefficients of polynomial determined.")
    return reg, poly, scaler

# Fit model
reg, poly, scaler = polynomial_regression(x1, x2, x3, x4, x5, y, DEG_ACTUAL)
print("scaler", scaler)



def denormalize(x):
    return x * np.array([10*101325, 125, 1.3, 0.7, 100]) + np.array([0, 10, 0, 0.9, 30])
def normalize(x):
    return (x - np.array([0, 10, 0, 0.9, 30])) * np.array([1/(10*101325), 1/125, 1/1.3, 1/0.7, 1/100])

def evaluate_poly(x):
    x = np.array([x])
    x = scaler.transform(x)
    x = poly.transform(x)
    return reg.predict(x)

def evaluate_poly_normalized(x_norm): 
    x_real = pd.Series(denormalize(x_norm))
    return evaluate_poly(x_real)

def numerical_gradient(x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (evaluate_poly_normalized(x_plus) - evaluate_poly_normalized(x_minus)) / (2 * h)
    return grad

def numerical_hessian(x, h=1e-5):
    n = len(x)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                x_plus = x.copy(); x_plus[i] += h
                x_minus = x.copy(); x_minus[i] -= h
                f_x_plus = evaluate_poly_normalized(x_plus)
                f_x = evaluate_poly_normalized(x)
                f_x_minus = evaluate_poly_normalized(x_minus)
                hessian[i, j] = (f_x_plus - 2*f_x + f_x_minus) / (h*h)
            else:
                x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
                x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
                x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
                x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
                hessian[i, j] = (
                    evaluate_poly_normalized(x_pp)
                    - evaluate_poly_normalized(x_pm)
                    - evaluate_poly_normalized(x_mp)
                    + evaluate_poly_normalized(x_mm)
                ) / (4 * h * h)
    return hessian

def is_maximum(hessian: np.ndarray) -> bool:
    return np.all(np.linalg.eigvals(hessian) < 0)

# --- Gradient Descent search --- (normalized gradient ~ 10*(1/0.5 * 6) ~ 10)
def find_high_tc_points(n_vars: int, n_attempts=500, tol=1, max_iter=1000):
    candidate_points = []
    X_samples = np.column_stack([x1, x2, x3, x4, x5])

    for attempt in range(n_attempts):
        i = np.random.randint(0, len(X_samples))
        x_start = normalize(X_samples[i])
        x_start[0] = random.uniform(0,1) #restricting to desired pressure range

        rel_deviation = 0.0000000001 #change to a value of your liking that still yields reasonable values
        deviation = rel_deviation*np.array([10*101325, 125, 1.3, 0.7, 100]) #for reasonable output, search around tested points

        x = x_start.copy()
        alpha = rel_deviation
        prev_grad_norm = float('inf')
        best_tc = evaluate_poly(pd.Series(denormalize(x)))

        for iteration in range(max_iter):
            grad = numerical_gradient(x)
            grad_norm = np.linalg.norm(grad)
            tc_current = evaluate_poly(pd.Series(denormalize(x)))

            x_new = x + alpha * grad

            # Clamp to both normalized bounds and 'neighborhood' range constraint
            x_new = np.clip(x_new, x_start - deviation, x_start + deviation)
            x_new = np.clip(x_new, 0, 1)
            tc_new = evaluate_poly(pd.Series(denormalize(x_new)))

            if tc_new > tc_current:
                x = x_new
                best_tc = tc_new
                alpha *= 1.05
            else:
                alpha *= 0.5

            alpha = np.clip(alpha, 1e-6, 0.1)
            prev_grad_norm = grad_norm

            if iteration == max_iter - 1 or grad_norm < tol:
                x_real = pd.Series(denormalize(x))
                tc_value = evaluate_poly(x_real)
                hessian = numerical_hessian(x)
                eigenvals = np.linalg.eigvals(hessian)
                neg_count = np.sum(eigenvals < 0)
                concavity_signs = np.sign(np.diag(hessian))

                #want either a local maximum in the majority of the variables or a neighborhood maximum via gradient descent
                if tc_value > 0 and ((not any(np.linalg.norm(x - p[0]) > tol for p in candidate_points)) or ((concavity_signs==-1).sum()>3)):
                    candidate_points.append([x_real.copy(), concavity_signs, eigenvals, tc_value, neg_count])
                    print(f"Appended point #{len(candidate_points)}")
                break

        if len(candidate_points) >= 50:
            break

    return candidate_points

# Run search and output results
critical_pts = find_high_tc_points(5)

for num,pt in enumerate(critical_pts):
    print(num+1, f"-th Critical point found at:\n {pt[0]}")
    print(f"Function value: {evaluate_poly(pt[0])}")
    print(f"Pressure: {pt[0][0]:.3f} Pa")
    print(f"Hessian diagonal signs: {pt[1]}")
    print(f"All eigenvalues negative (maximum): {np.all(pt[2] < 0)}\n")

rows = []
for i in range(len(critical_pts)): #coordinates format: [P AM S R EA]
    rows.append({
        'Modified Coordinates': str(critical_pts[i][0]),
        'T_c': evaluate_poly(critical_pts[i][0]),
        'Pressure': critical_pts[i][0][0],
        'Hessian_Signs': critical_pts[i][1],
        'Maximum in _/5 variables': (critical_pts[i][2] < 0).sum()
    })

with open('criticalpoints_experimentalV7.csv', 'w', newline='') as csvfile:
    variables = ['Modified Coordinates', 'T_c', 'Pressure', 'Hessian_Signs', 'Maximium in _/5 variables']
    writer = csv.DictWriter(csvfile, fieldnames=variables)
    writer.writeheader()
    writer.writerows(rows)
   
print(f"\n\n{len(critical_pts)} critical points (maxima) identified with pressure between 1-10 atm.")




'''
# test case
X_input_raw = np.array([[155000000000.0,32.065,0.0,0.87,201.5,14]])
X_input_scaled = scaler.transform(X_input_raw)
X_poly_input = poly.transform(X_input_scaled)
y_pred = reg.predict(X_poly_input)
print("Predicted T_exp:", y_pred)



#T_c Vague Estimation [Bonus]

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

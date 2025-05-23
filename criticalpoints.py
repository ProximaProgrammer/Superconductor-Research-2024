'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import csv


data = pd.read_csv('coefficientsV9.csv')
parameters = pd.read_csv('superconductorV7.csv')

Terms,coef = data['Term'][1:], data['Coefficient'][1:] #add constant terms back later

def evaluate_poly(x, terms):
    result = 0
    for coeff, term_str in terms:
        term_val = coeff*10**-30 #temporary reduction to prevent overflow errors
        for part in term_str.split():
            var = int(part[1:].split('^')[0])
            power = int(part.split('^')[1]) if '^' in part else 1
            term_val *= x.iloc[var] ** power
        result += term_val
    return result

def numerical_gradient(x: np.ndarray, terms: List[Tuple[float, str]], h=1e-5) -> np.ndarray:
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_plus.iloc[i] += h
        x_minus = x.copy()
        x_minus.iloc[i] -= h
        grad[i] = (evaluate_poly(x_plus, terms) - evaluate_poly(x_minus, terms)) / (2*h)
    return grad

def numerical_hessian(x: np.ndarray, terms: List[Tuple[float, str]], h=1e-5) -> np.ndarray:
    n = len(x)
    hessian = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                x_plus = x.copy()
                x_plus.iloc[i] += h
                x_minus = x.copy()
                x_minus.iloc[i] -= h
                hessian[i, j] = (evaluate_poly(x_plus, terms) - 2*evaluate_poly(x, terms) + evaluate_poly(x_minus, terms)) / (h*h)
            else:
                x_pp = x.copy()
                x_pp.iloc[i] += h
                x_pp.iloc[j] += h
                x_pm = x.copy()
                x_pm.iloc[i] += h
                x_pm.iloc[j] -= h
                x_mp = x.copy()
                x_mp.iloc[i] -= h
                x_mp.iloc[j] += h
                x_mm = x.copy()
                x_mm.iloc[i] -= h
                x_mm.iloc[j] -= h
                hessian[i, j] = (evaluate_poly(x_pp, terms) - evaluate_poly(x_pm, terms) - evaluate_poly(x_mp, terms) + evaluate_poly(x_mm, terms)) / (4*h*h)
    
    return hessian

def is_maximum(hessian: np.ndarray) -> bool:
    """Check if point is an approximate maximum using Hessian eigenvalues"""
    eigenvals = np.linalg.eigvals(hessian)
    return np.all(eigenvals < 0)

def find_high_tc_points(terms: List[Tuple[float, str]], n_vars: int, n_attempts=500, tol=1e-32, max_iter=1000) -> List[np.ndarray]:
    candidate_points = []
    
    for attempt in range(n_attempts):
        x = np.random.randn(n_vars) #initialize search with random coordinates to begin with
        x[4] = np.random.uniform(1*101325, 10*101325)  # Initialize pressure between 1-10 atm to search in that range (CHECK THIS)
        x = np.clip(x, [2800000000.0,91.48215,1.12704043108262,1.4653631578947401,76.8425,9], []) #for polynomial accuracy, restrict parameters to within a range based on real values

        alpha = 0.01
        prev_grad_norm = float('inf')
        
        for iteration in range(max_iter):
            grad = numerical_gradient(x, terms)
            grad_norm = np.linalg.norm(grad)
            
            # Adaptive learning rate instead of being 'stuck' computationally
            if grad_norm > prev_grad_norm:
                alpha *= 0.5
            else:
                alpha *= 1.05
            alpha = np.clip(alpha, 1e-6, 0.1)

            # If gradient is small enough, consider the point a critical point OR if we've converged to high T_c
            if grad_norm < tol or iteration == max_iter - 1:
                print("passed first level")
                # Ensure pressure constraint is satisfied
                if 1*101325<=x.iloc[0]<=10*101325:
                    # Check if it's a new point (allow more similar points for high Tc search)
                    if (np.all(x==x.clip())) and (not any(np.linalg.norm(x - p[0]) < tol*5 for p in candidate_points)):
                        tc_value = evaluate_poly(x, terms)
                        hessian = numerical_hessian(x, terms)
                        eigenvals = np.linalg.eigvals(hessian)
                        candidate_points.append([x.copy(), np.sign(np.diag(hessian)), eigenvals, tc_value])
                        print("appended point", len(candidate_points))
                break

            if len(candidate_points)==10: # to stop almost an indefinite search
                break
            
            # Gradient ascent (to find maxima)
            x_new = x + alpha * grad
            
            # Project pressure back to valid range
            x_new.iloc[0] = np.clip(x_new.iloc[0], 1*101325, 10*101325)
            
            x = x_new
            prev_grad_norm = grad_norm
    
    # Sort by T_c value and return top 10
    candidate_points.sort(key=lambda p: p[0], reverse=True)
    return candidate_points[:10]

Terms = [(coef[i],Terms[i]) for i in range(1,len(coef))]

critical_pts = find_high_tc_points(Terms, 6)

for num,pt in enumerate(critical_pts):
    print(num, f"-th Critical point found at:\n {pt[0]}")
    print(f"Function value: {evaluate_poly(pt[0], Terms)}")
    print(f"Pressure: {pt[0][0]:.3f} atm")
    print(f"Hessian diagonal signs: {pt[1]}")
    print(f"All eigenvalues negative (maximum): {np.all(pt[2] < 0)}\n")

rows = []
for i in range(len(critical_pts)): #coordinates format: [P AM S R EA]
    rows.append({
        'Modified Coordinates': str(critical_pts[i][0]), 
        'T_c*10^-30': evaluate_poly(critical_pts[i][0], Terms), 
        'Pressure': critical_pts[i][0][0],
        'Hessian_Signs': critical_pts[i][1],
        'Is_Maximum': np.all(critical_pts[i][2] < 0)
    })

# with open('criticalpoints_experimentalV7.csv', 'w', newline='') as csvfile:
#     variables = ['Modified Coordinates (w/ Electron Affinity', 'T_c*10^-30', 'Pressure', 'Hessian_Signs', 'Is_Maximum']
#     writer = csv.DictWriter(csvfile, fieldnames=variables)
#     writer.writeheader()
#     writer.writerows(rows)
    
print(f"\n\n{len(critical_pts)} critical points (maxima) identified with pressure between 1-10 atm.")
'''




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import csv

# --- Load data ---
data = pd.read_csv('coefficientsV9.csv')

def denormalize(x):
    return x * np.array([10, 125, 1.3, 0.7, 100]) + np.array([0, 10, 0, 0.9, 30])
def normalize(x):
    return (x - np.array([0, 10, 0, 0.9, 30])) * np.array([1/10, 1/125, 1/1.3, 1/0.7, 1/100])

Terms,coef = data['Term'][1:], data['Coefficient'][1:] #temporarily ignoring constant term

# --- Polynomial evaluation ---
def evaluate_poly(x, terms):
    result = coef.iloc[0] #adding constant term first

    for coeff, term_str in terms:
        term_val = coeff * 1e-30  # Scale down to avoid overflow
        print(coeff)
        print(term_str)
        for part in term_str.split():
            var = int(part[1:].split('^')[0])
            power = int(part.split('^')[1]) if '^' in part else 1
            term_val *= x.iloc[var] ** power
        result += term_val
    return result

def evaluate_poly(x, terms):
    result = 0
    for coeff, term_str in terms:
        term_val = coeff*10**-30 #temporary reduction to prevent overflow errors
        for part in term_str.split():
            var = int(part[1:].split('^')[0])
            power = int(part.split('^')[1]) if '^' in part else 1
            term_val *= x.iloc[var] ** power
        result += term_val
    return result


def evaluate_poly_normalized(x_norm, terms):
    x_real = pd.Series(denormalize(x_norm))
    return evaluate_poly(x_real, terms)

# --- Numerical tools ---
def numerical_gradient(x, terms, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (evaluate_poly_normalized(x_plus, terms) - evaluate_poly_normalized(x_minus, terms)) / (2 * h)
    return grad

def numerical_hessian(x, terms, h=1e-5):
    n = len(x)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                x_plus = x.copy(); x_plus[i] += h
                x_minus = x.copy(); x_minus[i] -= h
                f_x_plus = evaluate_poly_normalized(x_plus, terms)
                f_x = evaluate_poly_normalized(x, terms)
                f_x_minus = evaluate_poly_normalized(x_minus, terms)
                hessian[i, j] = (f_x_plus - 2*f_x + f_x_minus) / (h*h)
            else:
                x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
                x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
                x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
                x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
                hessian[i, j] = (
                    evaluate_poly_normalized(x_pp, terms)
                    - evaluate_poly_normalized(x_pm, terms)
                    - evaluate_poly_normalized(x_mp, terms)
                    + evaluate_poly_normalized(x_mm, terms)
                ) / (4 * h * h)
    return hessian

def is_maximum(hessian: np.ndarray) -> bool:
    return np.all(np.linalg.eigvals(hessian) < 0)

# --- Gradient ascent search --- (normalized gradient ~ 10^-30 * 10*(1/0.5 * 6) ~ 10^-29)
def find_high_tc_points(terms: List[Tuple[float, str]], n_vars: int, n_attempts=500, tol=1e-29, max_iter=1000):
    candidate_points = []
    for attempt in range(n_attempts):
        x = np.abs(np.random.rand(n_vars)) #generates entries randomly between 0 and 1
        #x = np.clip(x, [0,91.48215,1.12704043108262,1.4653631578947401,76.8425,9], []) #for polynomial accuracy, restrict parameters to within a range based on real values
      
        alpha = 0.01
        prev_grad_norm = float('inf')

        for iteration in range(max_iter):
            grad = numerical_gradient(x, terms)
            grad_norm = np.linalg.norm(grad)

            if grad_norm > prev_grad_norm:
                alpha *= 0.5
            else:
                alpha *= 1.05
            alpha = np.clip(alpha, 1e-6, 0.1)

            if grad_norm < tol or iteration == max_iter - 1:
                x_real = pd.Series(denormalize(x))
                tc_value = evaluate_poly(x_real, terms)
                hessian = numerical_hessian(x, terms)
                eigenvals = np.linalg.eigvals(hessian)
                if not any(np.linalg.norm(x - p[0]) < tol for p in candidate_points):
                    candidate_points.append([x_real.copy(), np.sign(np.diag(hessian)), eigenvals, tc_value])
                    print(f"Appended point #{len(candidate_points)}")
                break

            x_new = np.clip(x + alpha * grad, 0, 1)  # Keep normalized range
            x = x_new.clip(0, 1)  # clip afterward to [0,1] for safety
            prev_grad_norm = grad_norm

        if len(candidate_points) == 3: #eventually change from 3 to ≈20
            break

    #candidate_points.sort(key=lambda p: p.iloc[0], reverse=True)   #debug later
    return candidate_points[:3] #replace top 3 with top 10 or another num

# Run search and output results
critical_pts = find_high_tc_points(Terms, 5)

for num,pt in enumerate(critical_pts):
    print(num, f"-th Critical point found at:\n {pt[0]}")
    print(f"Function value: {evaluate_poly(pt[0], Terms)}")
    print(f"Pressure: {pt[0][0]:.3f} atm")
    print(f"Hessian diagonal signs: {pt[1]}")
    print(f"All eigenvalues negative (maximum): {np.all(pt[2] < 0)}\n")

rows = []
for i in range(len(critical_pts)): #coordinates format: [P AM S R EA]
    rows.append({
        'Modified Coordinates': str(critical_pts[i][0]), 
        'T_c*10^-30': evaluate_poly(critical_pts[i][0], Terms), 
        'Pressure': critical_pts[i][0][0],
        'Hessian_Signs': critical_pts[i][1],
        'Is_Maximum': np.all(critical_pts[i][2] < 0)
    })

# with open('criticalpoints_experimentalV7.csv', 'w', newline='') as csvfile:
#     variables = ['Modified Coordinates (w/ Electron Affinity', 'T_c*10^-30', 'Pressure', 'Hessian_Signs', 'Is_Maximum']
#     writer = csv.DictWriter(csvfile, fieldnames=variables)
#     writer.writeheader()
#     writer.writerows(rows)
    
print(f"\n\n{len(critical_pts)} critical points (maxima) identified with pressure between 1-10 atm.")

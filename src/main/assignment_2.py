import numpy as np
import scipy.linalg
import sympy as sp


def nevilles():
    x = [3.6, 3.8, 3.9]
    val = [1.675, 1.436, 1.318]
    neville = [[0] * len(x) for _ in range(len(x))]
    w = 3.7  

    for i in range(len(x)):
        neville[i][0] = val[i]

    for i in range(1, len(x)):
        for j in range(1, i + 1):
            term1 = (w - x[i - j]) * neville[i][j - 1]
            term2 = (w - x[i]) * neville[i - 1][j - 1]
            neville[i][j] = (term1 - term2) / (x[i] - x[i - j])  

    return neville[len(x) - 1][len(x) - 1]

def newtonForward():
    xi = [7.2, 7.4, 7.5, 7.6]
    yi = [23.5492, 25.3913, 26.8224, 27.4589]
    n = len(xi)
    
    diffs = [[0] * n for _ in range(n)]
    
    for i in range(n):
        diffs[i][0] = yi[i]
    
    for j in range(1, n):  
        for i in range(j, n): 
            diffs[i][j] = (diffs[i][j - 1] - diffs[i - 1][j - 1]) / (xi[i] - xi[i - j])

   
    print(diffs[1][1])
    print(diffs[2][2])
    print(diffs[3][3])
    
def newtonForwardInterpolation():
    xi = [7.2, 7.4, 7.5, 7.6]
    yi = [23.5492, 25.3913, 26.8224, 27.4589]
    n = len(xi)
    
    diffs = [[0] * n for _ in range(n)]
    
    for i in range(n):
        diffs[i][0] = yi[i]
    
    for j in range(1, n):  
        for i in range(j, n): 
            diffs[i][j] = (diffs[i][j - 1] - diffs[i - 1][j - 1]) / (xi[i] - xi[i - j])

    f_x0 = diffs[0][0]  
    f_x0x1 = diffs[1][1]  
    f_x0x1x2 = diffs[2][2]  
    f_x0x1x2x3 = diffs[3][3]  

    x = 7.3
    interpolated_value = (
        f_x0 +
        f_x0x1 * (x - xi[0]) +
        f_x0x1x2 * (x - xi[0]) * (x - xi[1]) +
        f_x0x1x2x3 * (x - xi[0]) * (x - xi[1]) * (x - xi[2])
    )

    print(interpolated_value)

def cubic_spline_interpolation():
    x = np.array([2, 5, 8, 10])
    f = np.array([3, 5, 7, 9])
    n = len(x)

    h = np.diff(x)  

    num_eqs = n - 2  

    A = np.zeros((n, n))
    b = np.zeros(n)

    A[0, 0] = 1
    A[-1, -1] = 1


    for i in range(1, num_eqs + 1):
        A[i, i - 1] = h[i - 1] 
        A[i, i] = 2 * (h[i - 1] + h[i])  
        A[i, i + 1] = h[i]  

        b[i] = (3 / h[i]) * (f[i + 1] - f[i]) - (3 / h[i - 1]) * (f[i] - f[i - 1])

        

    M = scipy.linalg.solve(A, b)

    print(A)
    print(b)
    print(M)

    return A, b, M


nevilles()
print()
newtonForward()
print()
newtonForwardInterpolation()
print()
cubic_spline_interpolation()



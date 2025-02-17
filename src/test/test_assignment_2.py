import pytest
import numpy as np
import scipy.linalg

from main import assignment_2

EXPECTED_NEVILLE = 1.5  
EXPECTED_NEWTON = 24.47 
EXPECTED_SPLINE_MATRIX = np.array([
    [1, 0, 0, 0],
    [3, 12, 3, 0],
    [0, 3, 12, 3],
    [0, 0, 0, 1]
])  
EXPECTED_SPLINE_RHS = np.array([0, 3.0, 3.0, 0])  
EXPECTED_SPLINE_SOLUTION = np.array([0, 1, 1, 0]) 

def test_nevilles():
    result = assignment_2.nevilles()
    assert pytest.approx(result, rel=1e-3) == EXPECTED_NEVILLE, f"Neville's result incorrect: {result}"

def test_newton_forward_interpolation():
    result = assignment_2.newtonForwardInterpolation()
    assert pytest.approx(result, rel=1e-3) == EXPECTED_NEWTON, f"Newton's interpolation incorrect: {result}"

def test_cubic_spline_interpolation():
    A, b, M = assignment_2.cubic_spline_interpolation()
    
    assert np.allclose(A, EXPECTED_SPLINE_MATRIX, atol=1e-3), "Spline matrix A incorrect"
    assert np.allclose(b, EXPECTED_SPLINE_RHS, atol=1e-3), "Spline RHS vector b incorrect"
    assert np.allclose(M, EXPECTED_SPLINE_SOLUTION, atol=1e-3), "Spline solution M incorrect"


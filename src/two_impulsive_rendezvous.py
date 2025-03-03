"""
Author: Ryan Gast
Date: 03/03/2025
Calculates the delta-v requirements for a 
two-impulse rendezvous using Clohessy-Wiltshire equations.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from rva_relative import rva_relative

@dataclass
class RendezvousResult:
    """Class for storing the results of a two-impulse rendezvous calculation."""
    delta_v0: np.ndarray      # First impulse delta-v vector (km/s)
    delta_vf: np.ndarray      # Second impulse delta-v vector (km/s)
    total_delta_v: float      # Magnitude of total delta-v (km/s)
    v0_plus: np.ndarray       # Velocity after first impulse (km/s)
    vf_minus: np.ndarray      # Velocity before second impulse (km/s)

def two_impulsive_rendezvous(rA, vA, rB, vB, tf, mu):
    """
    Calculate delta-v requirements for a two-impulse rendezvous using Clohessy-Wiltshire equations.
    
    Parameters:
    -----------
    rA : array_like
        Position vector of target satellite in geocentric frame (km)
    vA : array_like
        Velocity vector of target satellite in geocentric frame (km/s)
    rB : array_like
        Position vector of chaser satellite in geocentric frame (km)
    vB : array_like
        Velocity vector of chaser satellite in geocentric frame (km/s)
    tf : float
        Time of flight to rendezvous (seconds)
    mu : float
        Gravitational parameter (km^3/s^2)
    return_stm : bool, optional
        Whether to return state transition matrices (default: False)
    
    Returns:
    --------
    RendezvousResult
        Object containing rendezvous maneuver details
    """
    # Input validation
    if len(rA) != 3 or len(vA) != 3 or len(rB) != 3 or len(vB) != 3:
        raise ValueError("Position and velocity vectors must have 3 components")
    
    # Convert input arrays to numpy arrays if they aren't already
    rA = np.array(rA, dtype=float)
    vA = np.array(vA, dtype=float)
    rB = np.array(rB, dtype=float)
    vB = np.array(vB, dtype=float)
    
    if tf <= 0:
        raise ValueError("Time of flight must be positive")
    if mu <= 0:
        raise ValueError("Gravitational parameter must be positive")
    
    # Calculate relative position and velocity using rva_relative
    r0, v0_minus, _ = rva_relative(rA, vA, rB, vB, mu)
    
    # Calculate semi-major axis of the reference orbit
    a = np.linalg.norm(rA) / (2 - np.linalg.norm(vA)**2 * np.linalg.norm(rA) / mu)
    
    # Calculate mean motion
    n = np.sqrt(mu / a**3)
    
    # Compute STM components at time tf
    Phi_rr = compute_Phi_rr(n, tf)
    Phi_rv = compute_Phi_rv(n, tf)
    Phi_vr = compute_Phi_vr(n, tf)
    Phi_vv = compute_Phi_vv(n, tf)
    
    # Calculate v0_plus (velocity required after first impulse)
    v0_plus = -np.linalg.inv(Phi_rv) @ (Phi_rr @ r0)
    
    # Calculate first delta-v
    delta_v0 = v0_plus - v0_minus
    
    # Calculate velocity just before second impulse
    vf_minus = Phi_vr @ r0 + Phi_vv @ v0_plus
    
    # Second impulse needs to bring relative velocity to zero
    delta_vf = -vf_minus
    
    # Calculate total delta-v magnitude
    total_delta_v = np.linalg.norm(delta_v0) + np.linalg.norm(delta_vf)
    
    # Prepare result object
    return RendezvousResult(delta_v0, delta_vf, total_delta_v, v0_plus, vf_minus)

def compute_Phi_rr(n, t):
    """Compute the position-position state transition matrix."""
    nt = n * t
    cos_nt = np.cos(nt)
    sin_nt = np.sin(nt)
    
    return np.array([
        [4 - 3 * cos_nt, 0, 0],
        [6 * (sin_nt - nt), 1, 0],
        [0, 0, cos_nt]
    ])

def compute_Phi_rv(n, t):
    """Compute the position-velocity state transition matrix."""
    nt = n * t
    cos_nt = np.cos(nt)
    sin_nt = np.sin(nt)
    
    return np.array([
        [sin_nt / n, 2 * (1 - cos_nt) / n, 0],
        [2 * (cos_nt - 1) / n, (4 * sin_nt - 3 * nt) / n, 0],
        [0, 0, sin_nt / n]
    ])

def compute_Phi_vr(n, t):
    """Compute the velocity-position state transition matrix."""
    nt = n * t
    cos_nt = np.cos(nt)
    sin_nt = np.sin(nt)
    
    return np.array([
        [3 * n * sin_nt, 0, 0],
        [6 * n * (cos_nt - 1), 0, 0],
        [0, 0, -n * sin_nt]
    ])

def compute_Phi_vv(n, t):
    """Compute the velocity-velocity state transition matrix."""
    nt = n * t
    cos_nt = np.cos(nt)
    sin_nt = np.sin(nt)
    
    return np.array([
        [cos_nt, 2 * sin_nt, 0],
        [-2 * sin_nt, 4 * cos_nt - 3, 0],
        [0, 0, cos_nt]
    ])

# Example usage
if __name__ == "__main__":
    from sv_from_coe import sv_from_coe
    
    # Earth parameters
    mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
    earth_radius = 6378.0  # Earth's radius (km)
    
    # Test case with specified orbital parameters
    print("Test case with specified orbital parameters:")
    
    # Convert degrees to radians
    deg2rad = np.pi / 180
    
    # Satellite A: 300 km circular orbit
    alt_A = 300.0  # km
    r_A = earth_radius + alt_A  # km
    e_A = 0.0  # circular orbit
    theta_A = 60.0 * deg2rad  # true anomaly (rad)
    i_A = 40.0 * deg2rad  # inclination (rad)
    RAAN_A = 20.0 * deg2rad  # RA of ascending node (rad)
    omega_A = 0.0 * deg2rad  # argument of perigee (rad)
    
    # Calculate angular momentum for satellite A
    h_A = np.sqrt(mu * r_A)  # for circular orbit
    
    # Satellite B: elliptical orbit
    alt_perigee_B = 320.06  # km
    alt_apogee_B = 513.86  # km
    r_perigee_B = earth_radius + alt_perigee_B  # km
    r_apogee_B = earth_radius + alt_apogee_B  # km
    
    # Calculate orbital elements for satellite B
    a_B = (r_perigee_B + r_apogee_B) / 2  # semi-major axis (km)
    e_B = (r_apogee_B - r_perigee_B) / (r_apogee_B + r_perigee_B)  # eccentricity
    theta_B = 349.65 * deg2rad  # true anomaly (rad)
    i_B = 40.130 * deg2rad  # inclination (rad)
    RAAN_B = 19.819 * deg2rad  # RA of ascending node (rad)
    omega_B = 70.662 * deg2rad  # argument of perigee (rad)
    
    # Calculate angular momentum for satellite B
    h_B = np.sqrt(mu * a_B * (1 - e_B**2))
    
    # Convert to state vectors
    rA, vA = sv_from_coe([h_A, e_A, RAAN_A, i_A, omega_A, theta_A], mu)
    rB, vB = sv_from_coe([h_B, e_B, RAAN_B, i_B, omega_B, theta_B], mu)
    
    # Calculate rendezvous maneuver for 1 hour transfer
    tf = 3600 * 8 # 8 hour in seconds
    
    result = two_impulsive_rendezvous(rA, vA, rB, vB, tf, mu)
    
    print("\nSatellite A (Target) Orbital Parameters:")
    print(f"  Circular altitude: {alt_A:.2f} km")
    print(f"  Period: 1.5086 h")
    print(f"  True anomaly: {theta_A / deg2rad:.2f} deg")
    print(f"  Inclination: {i_A / deg2rad:.2f} deg")
    print(f"  RAAN: {RAAN_A / deg2rad:.2f} deg")
    print(f"  Argument of perigee: {omega_A / deg2rad:.2f} deg")
    
    print("\nSatellite B (Chaser) Orbital Parameters:")
    print(f"  Perigee altitude: {alt_perigee_B:.2f} km")
    print(f"  Apogee altitude: {alt_apogee_B:.2f} km")
    print(f"  Period: 1.5484 h")
    print(f"  True anomaly: {theta_B / deg2rad:.2f} deg")
    print(f"  Inclination: {i_B / deg2rad:.2f}")
    print(f"  RAAN: {RAAN_B / deg2rad:.2f} deg")
    print(f"  Argument of perigee: {omega_B / deg2rad:.2f} deg")
    
    print("\nState Vectors:")
    print(f"  rA = {rA} km")
    print(f"  vA = {vA} km/s")
    print(f"  rB = {rB} km")
    print(f"  vB = {vB} km/s")
    
    print("\nRendezvous Maneuver Results:")
    print(f"  Initial delta-v: {result.delta_v0} km/s (magnitude: {np.linalg.norm(result.delta_v0):.4f} km/s)")
    print(f"  Final delta-v: {result.delta_vf} km/s (magnitude: {np.linalg.norm(result.delta_vf):.4f} km/s)")
    print(f"  Total delta-v required: {result.total_delta_v:.4f} km/s")
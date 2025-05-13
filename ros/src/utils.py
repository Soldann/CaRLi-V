import numpy as np

def polar_to_cartesian(polar_points):
    """
    Converts Nx3 NumPy array of (r, theta, phi) to Cartesian coordinates (x, y, z).
    
    :param polar_points: NumPy array of shape (N, 3) with [r, theta, phi]
    :return: NumPy array of shape (N, 3) with [x, y, z]
    """
    r, theta, phi = polar_points[:, 0], polar_points[:, 1], polar_points[:, 2]
    
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(phi)
    
    return np.column_stack((x, y, z))

def cartesian_to_polar(points):
    """
    Converts Nx3 NumPy array of (x, y, z) points to polar coordinates (r, theta, phi).
    
    :param points: NumPy array of shape (N, 3) with [x, y, z]
    :return: NumPy array of shape (N, 3) with [r, theta, phi] (r in meters, theta/phi in radians)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # Azimuth angle
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))  # Elevation angle
    
    return np.column_stack((r, theta, phi))

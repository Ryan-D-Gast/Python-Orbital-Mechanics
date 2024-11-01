"""
Author: Ryan Gast
Date: 1/9/2024
return month and planet names based on month_id and planet_id.
    Converted From Matlab to Python from: 
    "Orbital Mechanics for Engineering Students" (2013), 
    Appendix D, D.34.
Version: 1.0
Contact: ryan.d.gast@gmail.com
"""

def month_planet_names(month_id, planet_id):
    """
    Returns the name of the month and planet based on the given month_id and planet_id.

    Parameters:
    month_id (int): The ID of the month (1-12).
    planet_id (int): The ID of the planet (1-9).

    Returns:
    tuple: A tuple containing the name of the month and planet.
    """
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 
               'Saturn', 'Uranus', 'Neptune', 'Pluto']
    month = months[month_id - 1]
    planet = planets[planet_id - 1]
    return month, planet
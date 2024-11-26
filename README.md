# Phonon Thermal Conductivity Calculator

This application calculates the phonon thermal conductivity of crystalline materials using the relaxation time approximation. It implements a simplified model that takes into account:

- Phonon group velocity
- Debye temperature
- Temperature dependence
- Simple relaxation time approximation

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

The main script `phonon_conductivity.py` contains a `PhononConductivity` class that handles the calculations. The example in the main function demonstrates calculations for Silicon, but you can modify the parameters for other materials.

To run the example:
```bash
python phonon_conductivity.py
```

### Parameters

- `temperature`: Temperature in Kelvin
- `lattice_constant`: Lattice constant in meters
- `group_velocity`: Group velocity of phonons in m/s
- `debye_temperature`: Debye temperature in Kelvin

### Features

1. Calculate thermal conductivity at a specific temperature
2. Plot thermal conductivity as a function of temperature
3. Customizable parameters for different materials

## Theory

The thermal conductivity is calculated using the kinetic theory approach with the relaxation time approximation. The implementation includes:

- Bose-Einstein distribution for phonon statistics
- Frequency-dependent relaxation time
- Integration over phonon frequencies up to the Debye frequency

## Example Output

The program will output:
1. The calculated thermal conductivity at the specified temperature
2. A plot showing the temperature dependence of thermal conductivity

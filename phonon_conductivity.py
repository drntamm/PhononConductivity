import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class MaterialProperties:
    """Data class for material properties"""
    name: str
    crystal_system: str
    lattice_constants: Tuple[float, ...]  # in meters
    sound_velocity: float  # in m/s
    debye_temperature: float  # in Kelvin
    atomic_mass: float  # in kg
    optical_modes: Dict[str, float]  # mode name -> frequency in THz
    description: str
    gruneisen_parameter: float  # Grüneisen parameter for anharmonicity
    point_defect_factor: float  # Point defect scattering strength
    umklapp_parameter: float  # Strength of Umklapp processes

class MaterialDatabase:
    """Database of material properties"""
    def __init__(self):
        self.materials = {
            'MgSiO3': MaterialProperties(
                name='MgSiO3',
                crystal_system='orthorhombic',
                lattice_constants=(4.775e-10, 4.932e-10, 6.902e-10),  # a, b, c
                sound_velocity=8000,
                debye_temperature=1017,
                atomic_mass=100.389 / constants.Avogadro,
                optical_modes={
                    'Si-O stretch': 25.0,
                    'O-Si-O bend': 15.0,
                    'Mg-O': 10.0,
                    'Rigid unit': 5.0
                },
                description='Perovskite structure (Pbnm space group)',
                gruneisen_parameter=1.5,  # Typical value for perovskites
                point_defect_factor=1e-4,
                umklapp_parameter=2.0
            ),
            'Al2O3': MaterialProperties(
                name='Al2O3',
                crystal_system='hexagonal',
                lattice_constants=(4.785e-10, 4.785e-10, 12.991e-10),  # a, a, c
                sound_velocity=10000,
                debye_temperature=1047,
                atomic_mass=101.961 / constants.Avogadro,
                optical_modes={
                    'E1(TO)': 7.8,
                    'E1(LO)': 10.2,
                    'A1(TO)': 11.0,
                    'A1(LO)': 12.5
                },
                description='Corundum structure (R-3c space group)',
                gruneisen_parameter=1.3,  # From literature
                point_defect_factor=5e-5,
                umklapp_parameter=1.8
            ),
            'MgO': MaterialProperties(
                name='MgO',
                crystal_system='cubic',
                lattice_constants=(4.212e-10,),  # a
                sound_velocity=7900,
                debye_temperature=945,
                atomic_mass=40.304 / constants.Avogadro,
                optical_modes={
                    'TO': 12.1,
                    'LO': 22.9
                },
                description='Rocksalt structure (Fm-3m space group)',
                gruneisen_parameter=1.45,  # From experimental data
                point_defect_factor=2e-5,
                umklapp_parameter=1.5
            ),
            'SrO': MaterialProperties(
                name='SrO',
                crystal_system='cubic',
                lattice_constants=(5.160e-10,),  # a
                sound_velocity=6500,  # Approximate value for rocksalt structure
                debye_temperature=470,  # From literature
                atomic_mass=103.62 / constants.Avogadro,
                optical_modes={
                    'TO': 7.32,  # Transverse optical mode frequency in THz
                    'LO': 12.1   # Longitudinal optical mode frequency in THz
                },
                description='Rocksalt structure (Fm-3m space group), wide-bandgap alkaline earth oxide',
                gruneisen_parameter=1.6,  # Estimated for ionic oxide
                point_defect_factor=3e-5,
                umklapp_parameter=1.7
            )
        }
    
    def get_material(self, name: str) -> MaterialProperties:
        """Get material properties by name"""
        return self.materials[name]
    
    def list_materials(self) -> List[str]:
        """List all available materials"""
        return list(self.materials.keys())

class PhononCalculator:
    """Calculator for phonon properties"""
    def __init__(self, material_name: str, temperature: float = 300):
        """
        Initialize calculator with material properties
        
        Args:
            material_name (str): Name of the material
            temperature (float): Temperature in Kelvin
        """
        self.db = MaterialDatabase()
        self.material = self.db.get_material(material_name)
        self.T = temperature
        
        # Physical constants
        self.k_B = constants.Boltzmann
        self.hbar = constants.hbar
        
        # Derived quantities
        self.omega_D = self.k_B * self.material.debye_temperature / self.hbar

    def get_dispersion_parameters(self, branch: str) -> Dict:
        """Get dispersion parameters based on material and branch type"""
        params = {
            'q_max': np.pi / self.material.lattice_constants[0],
            'v_s': self.material.sound_velocity
        }
        
        if self.material.crystal_system == 'cubic':
            params.update({
                'LA_factor': 1.0,
                'TA_factor': 0.5,
                'optical_dispersion': 0.1
            })
        elif self.material.crystal_system == 'hexagonal':
            params.update({
                'LA_factor': 1.0,
                'TA_factor': [0.6, 0.5],  # Different TA modes
                'optical_dispersion': 0.15
            })
        else:  # orthorhombic
            params.update({
                'LA_factor': 1.2,  # Enhanced due to anisotropy
                'TA_factor': [0.6, 0.5],  # Different TA modes
                'optical_dispersion': {
                    'stretch': 0.05,
                    'bend': 0.1,
                    'other': 0.15
                }
            })
        return params

    def dispersion_relation(self, q: np.ndarray, branch: str) -> np.ndarray:
        """Calculate phonon dispersion relation"""
        params = self.get_dispersion_parameters(branch)
        q_max = params['q_max']
        q_abs = np.abs(q)
        
        if branch == 'LA':
            base = params['v_s'] * q_max * np.sin(q_abs/q_max)
            if self.material.crystal_system == 'orthorhombic':
                return base * (1 + 0.2 * np.cos(2*q_abs/q_max))
            return base
        
        elif branch in ['TA1', 'TA2']:
            idx = 0 if branch == 'TA1' else 1
            if isinstance(params['TA_factor'], list):
                factor = params['TA_factor'][idx]
            else:
                factor = params['TA_factor']
            return factor * params['v_s'] * q_max * np.sin(q_abs/q_max)
        
        elif branch in self.material.optical_modes:
            base_freq = self.material.optical_modes[branch] * 2 * np.pi * 1e12
            if isinstance(params['optical_dispersion'], dict):
                if 'stretch' in branch.lower():
                    disp = params['optical_dispersion']['stretch']
                elif 'bend' in branch.lower():
                    disp = params['optical_dispersion']['bend']
                else:
                    disp = params['optical_dispersion']['other']
            else:
                disp = params['optical_dispersion']
            return base_freq - disp * base_freq * np.cos(q_abs/q_max)
        
        else:
            raise ValueError(f"Unknown branch: {branch}")

    def plot_phonon_spectrum(self, n_zones: int = 3):
        """Plot phonon dispersion and DOS with interactive zone control"""
        q_points = 400
        dos_points = 1000
        
        # Create figure with subplots and slider
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 0.1], width_ratios=[2, 1])
        ax_disp = plt.subplot(gs[0, 0])
        ax_dos = plt.subplot(gs[0, 1], sharey=ax_disp)
        ax_slider = plt.subplot(gs[1, 0])
        
        def plot_dispersion(n_zones):
            ax_disp.clear()
            
            # Generate extended q points
            q_base = np.linspace(-np.pi/self.material.lattice_constants[0],
                               np.pi/self.material.lattice_constants[0], q_points)
            q_extended = np.array([])
            labels_extended = []
            positions_extended = []
            
            # Create periodic extension
            for i in range(-n_zones//2, n_zones//2 + 1):
                q_shifted = q_base + i * 2*np.pi/self.material.lattice_constants[0]
                q_extended = np.append(q_extended, q_shifted)
                
                if i == 0:
                    labels = ['Γ', 'X', 'M', 'Γ']
                else:
                    labels = [f'Γ{i}', f'X{i}', f'M{i}', f'Γ{i}']
                
                positions = np.linspace(q_shifted[0], q_shifted[-1], len(labels))
                labels_extended.extend(labels)
                positions_extended.extend(positions)
            
            # Plot acoustic modes
            freq_LA = self.dispersion_relation(q_extended, 'LA') / (2 * np.pi * 1e12)
            ax_disp.plot(q_extended, freq_LA, 'b-', label='LA', linewidth=2)
            
            # Plot TA modes based on crystal system
            if self.material.crystal_system in ['hexagonal', 'orthorhombic']:
                freq_TA1 = self.dispersion_relation(q_extended, 'TA1') / (2 * np.pi * 1e12)
                freq_TA2 = self.dispersion_relation(q_extended, 'TA2') / (2 * np.pi * 1e12)
                ax_disp.plot(q_extended, freq_TA1, 'r--', label='TA1', linewidth=2)
                ax_disp.plot(q_extended, freq_TA2, 'r:', label='TA2', linewidth=2)
            else:
                freq_TA = self.dispersion_relation(q_extended, 'TA1') / (2 * np.pi * 1e12)
                ax_disp.plot(q_extended, freq_TA, 'r--', label='TA', linewidth=2)
            
            # Plot optical modes
            colors = ['g', 'm', 'c', 'y']
            for i, mode in enumerate(self.material.optical_modes):
                freq_opt = self.dispersion_relation(q_extended, mode) / (2 * np.pi * 1e12)
                ax_disp.plot(q_extended, freq_opt, f'{colors[i%len(colors)]}-',
                           label=mode, linewidth=2)
            
            # Add zone boundaries
            for i in range(-n_zones//2, n_zones//2 + 2):
                x = i * 2*np.pi/self.material.lattice_constants[0]
                ax_disp.axvline(x=x, color='k', linestyle='--', alpha=0.3)
            
            ax_disp.set_xticks(positions_extended)
            ax_disp.set_xticklabels(labels_extended)
            ax_disp.set_xlabel('Wave Vector')
            ax_disp.set_ylabel('Frequency (THz)')
            ax_disp.set_title(f'Phonon Dispersion - {self.material.name}\n{self.material.description}')
            ax_disp.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
            ax_disp.grid(True, alpha=0.3)
        
        # Calculate DOS
        q = np.linspace(-np.pi/self.material.lattice_constants[0],
                       np.pi/self.material.lattice_constants[0], q_points)
        
        # Collect all frequencies
        all_freqs = []
        # Acoustic modes
        all_freqs.append(self.dispersion_relation(q, 'LA') / (2 * np.pi * 1e12))
        if self.material.crystal_system in ['hexagonal', 'orthorhombic']:
            all_freqs.extend([
                self.dispersion_relation(q, 'TA1') / (2 * np.pi * 1e12),
                self.dispersion_relation(q, 'TA2') / (2 * np.pi * 1e12)
            ])
        else:
            all_freqs.append(self.dispersion_relation(q, 'TA1') / (2 * np.pi * 1e12))
        
        # Optical modes
        for mode in self.material.optical_modes:
            all_freqs.append(self.dispersion_relation(q, mode) / (2 * np.pi * 1e12))
        
        # Calculate DOS
        freq_max = max(np.max(freqs) for freqs in all_freqs)
        frequencies = np.linspace(0, freq_max * 1.1, dos_points)
        dos = np.zeros_like(frequencies)
        sigma = 0.3
        
        for mode_freqs in all_freqs:
            for freq in mode_freqs:
                dos += np.exp(-(frequencies - freq)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        
        # Plot DOS
        ax_dos.plot(dos, frequencies, 'k-', linewidth=2)
        ax_dos.fill_betweenx(frequencies, 0, dos, alpha=0.3)
        ax_dos.set_xlabel('DOS (arb. units)')
        ax_dos.set_title('Density of States')
        ax_dos.grid(True, alpha=0.3)
        ax_dos.set_yticklabels([])
        
        # Create slider
        slider = Slider(ax_slider, 'Number of Brillouin Zones', 1, 7,
                       valinit=n_zones, valstep=2)
        
        def update(val):
            n = int(slider.val)
            plot_dispersion(n)
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        plot_dispersion(n_zones)
        
        plt.tight_layout()
        plt.show()

    def calculate_mean_free_path(self, temperature: float) -> float:
        """Calculate phonon mean free path with temperature dependence"""
        # Base mean free path at room temperature (300K)
        mfp_300k = 10.0  # nanometers
        
        # Temperature scaling (phonon-phonon scattering dominates at high T)
        # At high temperatures, mean free path scales as 1/T
        scaled_mfp = mfp_300k * (300.0 / temperature)
        
        # Apply anharmonic correction
        anharmonic_factor = 1.0 / (1.0 + self.material.gruneisen_parameter * \
                                  (temperature / self.material.debye_temperature))
        
        return scaled_mfp * anharmonic_factor

    def calculate_thermal_conductivity(self, temperature: float) -> float:
        """Calculate thermal conductivity with anharmonic corrections"""
        # Debug prints
        print(f"\nDebug info for {self.material.name} at {temperature}K:")
        
        # Unit conversion factor (to get W/m·K)
        unit_factor = 1.0  # Base units are already in W/m·K
        
        # Volume per unit cell in m³
        if self.material.crystal_system == 'cubic':
            volume = self.material.lattice_constants[0]**3
        elif self.material.crystal_system == 'hexagonal':
            volume = np.sqrt(3)/2 * self.material.lattice_constants[0]**2 * self.material.lattice_constants[2]
        else:  # orthorhombic
            volume = np.prod(self.material.lattice_constants)
        
        print(f"Volume: {volume:e} m³")
        
        # Specific heat with anharmonic correction (J/m³·K)
        if temperature > self.material.debye_temperature:
            # High-T limit with anharmonic correction
            cv_harmonic = 3 * self.k_B / volume
            anharmonic_factor = 1 + self.material.gruneisen_parameter * \
                              (temperature / self.material.debye_temperature)
        else:
            # Debye model with anharmonic correction
            x = self.material.debye_temperature / temperature
            cv_harmonic = 9 * self.k_B * (temperature/self.material.debye_temperature)**3 * \
                         self.debye_integral(x) / volume
            anharmonic_factor = 1 + 0.5 * self.material.gruneisen_parameter * \
                              (temperature / self.material.debye_temperature)**2
        
        # Apply anharmonic correction to specific heat
        cv = cv_harmonic * anharmonic_factor * 1e6  # Convert to J/m³·K
        print(f"Specific heat: {cv:e} J/(m³·K)")
        
        # Get anharmonic-corrected mean free path (in meters)
        mfp = self.calculate_mean_free_path(temperature) * 1e-9  # Convert to meters
        print(f"Mean free path: {mfp:e} m")
        print(f"Sound velocity: {self.material.sound_velocity:e} m/s")
        
        # Calculate thermal conductivity with mode-specific contributions
        k_acoustic = unit_factor * cv * self.material.sound_velocity * mfp / 3.0
        print(f"Acoustic contribution: {k_acoustic:.2f} W/(m·K)")
        
        # Add optical mode contributions (typically 10-20% of acoustic contribution)
        optical_factor = 0.15  # 15% contribution from optical modes
        if temperature > self.material.debye_temperature / 2:
            k_optical = k_acoustic * optical_factor * \
                       np.exp(-self.material.debye_temperature / (2 * temperature))
            print(f"Optical contribution: {k_optical:.2f} W/(m·K)")
        else:
            k_optical = 0  # Negligible optical contribution at low T
            print("Optical contribution: negligible at low T")
        
        total_k = max(k_acoustic + k_optical, 1e-10)  # Ensure non-negative conductivity
        print(f"Total thermal conductivity: {total_k:.2f} W/(m·K)")
        return total_k

    def debye_integral(self, x: float) -> float:
        """Calculate Debye integral for specific heat"""
        def integrand(t):
            return t**4 * np.exp(t) / (np.exp(t) - 1)**2
        
        from scipy import integrate
        result, _ = integrate.quad(integrand, 0, x)
        return result

    def plot_thermal_conductivity(self, t_min: float = 300, t_max: float = 3000, n_points: int = 100):
        """Plot thermal conductivity as a function of temperature"""
        temperatures = np.linspace(t_min, t_max, n_points)
        conductivities = np.array([self.calculate_thermal_conductivity(t) for t in temperatures])
        
        plt.figure(figsize=(10, 6))
        plt.plot(temperatures, conductivities, '-', linewidth=2)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Thermal Conductivity (W/m·K)')
        plt.title(f'Thermal Conductivity - {self.material.name}\n{self.material.description}')
        plt.grid(True, alpha=0.3)
        plt.show()

def plot_thermal_conductivity_comparison(materials, temperatures):
    """Create a scatter plot comparing thermal conductivities of different materials"""
    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', 'D', '^']  # Different markers for each material
    colors = ['b', 'r', 'g', 'm']   # Different colors for each material
    
    # Experimental data from literature
    exp_data = {
        'MgSiO3': {  # Values from various studies on perovskite MgSiO3
            'temps': [300, 500, 1000],
            'k_values': [5.0, 3.2, 2.1],  # W/(m·K)
            'uncertainties': [0.5, 0.4, 0.3]
        },
        'Al2O3': {   # Values for single crystal alumina
            'temps': [300, 500, 1000],
            'k_values': [30.0, 15.0, 8.0],
            'uncertainties': [2.0, 1.5, 1.0]
        },
        'MgO': {     # Values for single crystal MgO
            'temps': [300, 500, 1000],
            'k_values': [60.0, 25.0, 12.0],
            'uncertainties': [5.0, 3.0, 2.0]
        },
        'SrO': {     # Values from studies on crystalline SrO
            'temps': [300, 500, 1000],
            'k_values': [12.0, 7.0, 4.0],
            'uncertainties': [1.0, 0.8, 0.6]
        }
    }
    
    for idx, material in enumerate(materials):
        # Plot calculated values
        calc = PhononCalculator(material)
        conductivities = [calc.calculate_thermal_conductivity(t) for t in temperatures]
        
        plt.scatter(temperatures, conductivities, 
                   label=f'{material} (Calculated)',
                   marker=markers[idx],
                   c=colors[idx],
                   s=100,
                   alpha=0.7)
        
        # Add connecting lines for calculated values
        plt.plot(temperatures, conductivities,
                c=colors[idx],
                linestyle='--',
                alpha=0.3)
        
        # Plot experimental values if available
        if material in exp_data:
            exp = exp_data[material]
            plt.errorbar(exp['temps'], exp['k_values'],
                        yerr=exp['uncertainties'],
                        label=f'{material} (Experimental)',
                        fmt=markers[idx],
                        c=colors[idx],
                        markersize=10,
                        capsize=5,
                        alpha=1.0,
                        markerfacecolor='none')
    
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Thermal Conductivity (W/m·K)', fontsize=12)
    plt.title('Thermal Conductivity: Calculated vs Experimental', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add minor grid for better readability
    plt.grid(True, which='minor', alpha=0.1)
    plt.minorticks_on()
    
    # Use log scale for y-axis due to large range
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

def main():
    # Create calculator for different materials
    materials = MaterialDatabase().list_materials()
    print("Available materials:", materials)
    
    # Define temperatures for comparison
    key_temps = [300, 500, 1000, 2000, 3000]
    
    # Create scatter plot comparison
    plot_thermal_conductivity_comparison(materials, key_temps)
    
    # Calculate individual properties
    for material in materials:
        print(f"\nCalculating properties for {material}...")
        calc = PhononCalculator(material)
        
        # Calculate and print thermal conductivity at key temperatures
        print(f"\nThermal Conductivity for {material}:")
        for temp in key_temps:
            k = calc.calculate_thermal_conductivity(temp)
            print(f"At {temp}K: {k:.2f} W/(m·K)")
        
        # Plot thermal conductivity
        calc.plot_thermal_conductivity()
        
        # Plot phonon spectrum
        calc.plot_phonon_spectrum()

calc = PhononCalculator("SrO")
calc.plot_phonon_spectrum()
if __name__ == "__main__":
    main()

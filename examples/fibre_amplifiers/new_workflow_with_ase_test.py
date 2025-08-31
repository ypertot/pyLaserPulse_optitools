#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NOTE: This is an example script to demonstrate the new four-stage workflow.
# Due to persistent environment issues (segmentation faults), this script
# could not be tested. It is provided to show the intended usage of the new
# features. The plotting part is commented out as it was identified as the
# source of the crash in the provided environment.

from pyLaserPulse import grid
from pyLaserPulse import pulse
from pyLaserPulse.solver import Solver
import pyLaserPulse.catalogue_components.active_fibres as af
# from pyLaserPulse import single_plot_window
import numpy as np


#############################################
# Choose a directory for saving the data.   #
# Leave as None if no data should be saved. #
#############################################
directory = None

############################################################
# Set time-frequency grid, pulse, and component parameters #
############################################################

# Time-frequency grid parameters
points = 2**9         # Number of grid points
central_wl = 1030e-9  # Central wavelength, m
max_wl = 1200e-9      # Maximum wavelength, m

# Laser pulse parameters
tau = 150e-15         # Pulse duration, s
P_peak = [150, .15]   # [P_x, P_y], W
f_rep = 40e6          # Repetition frequency, Hz
shape = 'sech'        # Can also take 'Gauss'

# Yb-fibre parameters
L = 1                                # length, m
ase_points = 2**8                    # number of points in pump & ASE grid
ase_wl_lims = [900e-9, max_wl]       # wavelength limits for ASE grid
bounds = {'co_pump_power': 0, 'counter_pump_power': 0} # No pump in this stage

# New model parameters
pump_power = 1.0 # W
pump_duration = 1e-3 # s
delay_time = 1e-5 # s

##############################################################
# Instantiate the time-frequency grid, pulse, and components #
##############################################################
print("Initializing grid, pulse, and fiber...")
# Time-frequency grid defined using the grid module
g = grid.grid(points, central_wl, max_wl)

# pulse defined using the pulse module
p = pulse.pulse(tau, P_peak, shape, f_rep, g)
initial_pulse_energy = p.pulse_energy

# Nufern PM-YSF-HI-HP defined using the catalogue_components module
# Note: boundary_conditions are simplified for the new model in this example
ydf = af.Nufern_PM_YSF_HI_HP(g, L, p.repetition_rate, ase_points, ase_wl_lims,
                             bounds)
print("Initialization complete.")
################################################################
# Execute the new four-stage process                           #
################################################################

# Stage 1: Pumping and ASE Seeding
print("Stage 1: Pumping and ASE Seeding...")
ydf.calculate_inversion_from_pump(pump_power=pump_power, pump_duration=pump_duration)
print("Pumping and ASE seeding complete.")

# Stage 2: Decay and ASE Amplification
print("Stage 2: Decay and ASE Amplification...")
ydf.amplify_ase_during_decay(delay_time=delay_time)
print("Decay and ASE amplification complete.")

# Stage 3: Signal Amplification
print("Stage 3: Signal Amplification...")
print("Initializing solver...")
solver = Solver(p, ydf)
print("Solver initialized.")
print("Solving...")
final_pulse = solver.solve()
print("Solve complete.")

######################
#    Check output    #
######################
print("Initial pulse energy: ", initial_pulse_energy)
print("Final pulse energy: ", final_pulse.pulse_energy)

# A simple check to see if the pulse was amplified
if final_pulse.pulse_energy > initial_pulse_energy:
    print("Pulse was amplified successfully.")
else:
    print("Pulse was not amplified.")

# Optional: Plotting
# try:
#     from pyLaserPulse import single_plot_window
#     from pyLaserPulse.utils import get_ESD_and_PSD
#     final_pulse.get_ESD_and_PSD(g, final_pulse.field)

#     plot_dict = {
#         'name': 'Test plot',
#         'is_toplevel': True,
#         'pulse_obj': final_pulse,
#         'grid_obj': g,
#         'B_integral_samples': [],
#         'field_samples': [],
#         'sample_points': [],
#         'PSD_at_output': final_pulse.power_spectral_density,
#         'ESD_at_output': final_pulse.energy_spectral_density,
#         'time_domain_at_output': np.sum(np.abs(final_pulse.field)**2, axis=0),
#     }

#     single_plot_window.matplotlib_gallery.launch_plot(plot_dicts=[plot_dict])
# except Exception as e:
#     print(f"Plotting failed: {e}")

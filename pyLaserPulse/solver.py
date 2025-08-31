import numpy as np
from .pulse import Pulse
from .abstract_bases import active_fibre_base
import scipy.constants as const
from . import utils

class Solver:
    def __init__(self, pulse: Pulse, fiber: active_fibre_base):
        self.pulse = pulse
        self.fiber = fiber
        # Make a copy so the original fiber object is not modified
        self.N2_profile = np.copy(getattr(fiber, 'N2_profile', None))
        if self.N2_profile is None:
            raise ValueError("The fiber object must have an N2_profile attribute. Please run calculate_inversion_from_pump() first.")

    def _nonlinear_step(self, field, dz, pm):
        """
        Apply and return nonlinear contribution to RK4IP step
        """
        if np.any(np.isnan(field)):
            return np.ones_like(field) * np.nan
        else:
            # _r indicates switch of pol. axes, i.e., [x, y] --> [y, x]
            field_r = field[::-1, :]
            P = field.real**2 + field.imag**2
            P_r = P[::-1, :]
            P_fft = utils.fft(P)
            P_fft_r = P_fft[::-1, :]
            conjfield = np.conj(field)
            conjfield_r = conjfield[::-1, :]
            conjfield_pm = utils.ifft(utils.fft(conjfield) * pm)

            SPM_XPM_DFWM = field * (1. - self.fiber.fR) * (P + (2. / 3.) * P_r) \
                + (1. - self.fiber.fR) * field_r**2 * conjfield_pm / 3.
            Raman_SPM_XPM = self.fiber.fR * field * self.pulse.grid.dt * utils.ifft(
                (self.fiber.Raman[:, 0] + self.fiber.Raman[:, 1]) * P_fft + self.fiber.Raman[:, 0] * P_fft_r)
            Raman_DFWM = self.fiber.fR * field_r * self.pulse.grid.dt * \
                utils.ifft(0.5 * self.fiber.Raman[:, 1] * utils.fft(
                    field * conjfield_r + field_r * conjfield_pm))
            k = self.fiber.self_steepening * dz * utils.fft(
                SPM_XPM_DFWM + Raman_SPM_XPM + Raman_DFWM)
            return k

    def _RK4IP(self, phasematching, dz, field_spec):
        """
        Apply the linear and nonlinear contributions over the propagation step
        using the Runge-Kutte 4th-order interaction picture method.
        """
        if np.any(np.isnan(field_spec)):
            return np.ones_like(field_spec) * np.nan
        else:
            uu1 = utils.ifft(field_spec)
            half_step = np.exp(-0.5 * self.fiber.linear_operator * dz)

            uip = half_step * field_spec

            k1 = self._nonlinear_step(uu1, dz, phasematching)
            k1 *= half_step

            uu2 = utils.ifft(uip + 0.5 * k1)
            k2 = self._nonlinear_step(uu2, dz, phasematching)

            uu3 = utils.ifft(uip + 0.5 * k2)
            k3 = self._nonlinear_step(uu3, dz, phasematching)

            uu4 = utils.ifft(half_step * (uip + k3))
            k4 = self._nonlinear_step(uu4, dz, phasematching)

            return half_step * (uip + k1 / 6. + k2 / 3. + k3 / 3.) + k4 / 6.

    def solve(self):
        dz = self.pulse.dz
        num_steps = int(self.fiber.L / dz)

        # Ensure N2_profile has the same number of steps
        if len(self.N2_profile) != num_steps:
             self.N2_profile = np.interp(np.linspace(0, self.fiber.L, num_steps), np.linspace(0, self.fiber.L, len(self.N2_profile)), self.N2_profile)

        z_grid = np.linspace(0, self.fiber.L, num_steps)
        ufft = utils.fft(self.pulse.field, axis=-1)

        for i in range(num_steps):
            z = z_grid[i]

            # 1. Calculate Gain
            N2 = self.N2_profile[i]
            N1 = self.fiber.N_tot - N2

            gain_coefficient = self.fiber.signal_overlaps * (self.fiber.signal_emission_cs * N2 - self.fiber.signal_absorption_cs * N1)

            # 2. Apply Gain (in time domain)
            A_time = utils.ifft(ufft)

            energy_before = np.sum(np.abs(A_time)**2) * self.pulse.grid.dt

            # Apply gain for half step
            A_time *= np.exp(0.5 * gain_coefficient * dz)

            energy_after = np.sum(np.abs(A_time)**2) * self.pulse.grid.dt

            ufft = utils.fft(A_time)

            # 3. Deplete Inversion
            delta_energy = energy_after - energy_before
            if delta_energy > 0:
                photons_extracted = delta_energy / (const.h * self.pulse.grid.v_c)

                # Volume of the gain medium slice
                mode_area = self.fiber.signal_mode_area[self.pulse.grid.midpoint]
                volume = mode_area * dz

                delta_N2 = photons_extracted / volume
                self.N2_profile[i] = max(0, N2 - delta_N2)

            # 4. Propagate
            phasematching = self.fiber._DFWM_phasematching(z)
            ufft = self._RK4IP(phasematching, dz, ufft)

            # Second half of gain application
            A_time = utils.ifft(ufft)
            A_time *= np.exp(0.5 * gain_coefficient * dz)
            ufft = utils.fft(A_time)


        self.pulse.field = utils.ifft(ufft, axis=-1)
        return self.pulse

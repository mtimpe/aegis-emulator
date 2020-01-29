# coding: utf-8
import numpy as np


""" Semi-analytic model for perfectly inelastic mergers. """


__author__ = "Miles Timpe, Maria Han Veiga, and Mischa Knabenhans"
__maintainer__ = "Miles Timpe"
__credits__ = ["Miles Timpe", "Maria Han Veiga", "Mischa Knabenhans"]
__email__ = "mtimpe@physik.uzh.ch"
__copyright__ = "Copyright 2020, ICS"
__license__ = "GNU General Public License v3.0"
__version__ = "0.1.0"
__status__ = "Development"


class PIMEmulator:
    """Perfectly inelastic merger."""

    def __init__(self, data, output_dir):

        super().__init__(data, output_dir)

        self.name = type(self).__name__


    def expected_breakup(rho_bulk):
        """Calculates the expected breakup rate for a homogeneous body using
           MacLaurin's formula.

        Args:
            rho_bulk (float): Bulk density of planet in g/cm3.

        Returns:
            float: Theoretical breakup rotation rate in Hz.
        """
        # Critical value of h for a homogeneous axial ellipsiod
        h_crit = 0.44931

        omega2 = np.pi * G_CGS * rho_bulk * h_crit

        return np.sqrt(omega2)


    def predict(self):

        # Critical value of h for a homogeneous axial ellipsiod
        h_crit = 0.44931

        # Newton's gravitational constant
        G_EES = 1.536e-6     # RE3 ME-1 s-2

        # Material densities
        gcm3_to_ee3 = 0.043444993

        rho_gran = 2.70 * gcm3_to_ee3    # Earth masses per cubic Earth radii
        rho_iron = 7.86 * gcm3_to_ee3   # Earth masses per cubic Earth radii


        for idx, collision in data.x.iterrows():

            mtotal = collision['mtotal']
            gamma  = collision['gamma']
            b_inf  = collision['b_inf']
            v_inf  = collision['v_inf']

            targ_core = collision['targ_core']
            targ_omega_norm = collision['targ_omega_norm']
            targ_theta = collision['targ_theta']
            targ_phi = collision['targ_phi']

            proj_core = collision['proj_core']
            proj_omega_norm = collision['proj_omega_norm']
            proj_theta = collision['proj_theta']
            proj_phi = collision['proj_phi']


            # Masses [Earth masses]
            targ_mass = mtotal / (1. + gamma)
            proj_mass = mtotal * gamma / (1. + gamma)

            # Bulk densities [g/cm3]
            targ_rho_bulk = rho_iron * targ_core + rho_gran * (1. - targ_core)
            proj_rho_bulk = rho_iron * proj_core + rho_gran * (1. - proj_core)

            # Radii [Earth radii]
            targ_radius = (3 * targ_mass / (4 * np.pi * targ_rho_bulk))**(1/3)
            proj_radius = (3 * proj_mass / (4 * np.pi * proj_rho_bulk))**(1/3)

            # Critical radius [Earth radii]
            R_crit = targ_radius + proj_radius

            # Rotation rates
            targ_omega_crit = maclaurin_limit(targ_rho_bulk)  # Hz
            proj_omega_crit = maclaurin_limit(proj_rho_bulk)  # Hz

            targ_omega = targ_omega_norm * targ_omega_crit  # Hz
            proj_omega = proj_omega_norm * proj_omega_crit  # Hz

            # Moments of inertia
            I_targ = (2/5) * targ_mass * targ_radius**2
            I_proj = (2/5) * proj_mass * proj_radius**2

            # Angular momentum
            Lmag_targ = I_targ * omega_targ
            Lmag_proj = I_proj * omega_proj

            # Angular momentum vectors
            Lvec_targ = np.array(Lmag_targ * Uvec_targ)
            Lvec_proj = np.array(Lmag_proj * Uvec_proj)

            # Two-body escape velocity (Earth radii per second)
            v_esc_tot = np.sqrt(2. * G_EES * self.mtotal / R_crit)

            # Target escape velocity (Earth radii per second)
            v_esc_targ = np.sqrt(2. * G_EES * targ_mass / R_crit)

            # Maximum impact parameter (Earth radii)
            b_grav = R_crit * np.sqrt(1. + (1. / v_inf**2.))

            # Convert input parameters to sensible units
            b_inf *= R_crit      # Earth radii
            v_inf *= v_esc_targ  # Earth radii per second


            # Orbital angular momentum
            L_orb = proj_mass * b_inf * v_inf

            # Core fraction of largest remnant
            Fe_lr = (targ_mass * targ_core + proj_mass * proj_core) / mtotal


            # Bulk density of largest remnant
            rho_lr = 7.86 * Fe_lr + 2.7 * (1 - Fe_lr)  # g/cm3
            rho_lr *= gcm3_to_ee3  # Earth masses per cubic Earth radii


            # Radius of largest remnant
            R_lr = (3. * mtotal / (4. * np.pi * rho_lr))**(1./3.)  # Earth radii


            # Moment of inertia (Earth masses per squared Earth radii)
            I_lr = (2./5.) * mtotal * R_lr**2.


            # Expected breakup rate in Hertz
            omega_crit = np.sqrt(np.pi * G_EES * rho_lr * h_crit)


            # Orbital angular momentum vector
            Lvec_orb = [0, 0, L_orb]

            Lvec_lr = Lvec_targ + Lvec_proj + Lvec_orb

            Lmag_lr = np.linalg.norm(Lvec_lr)

            omega_lr = abs(Lmag_lr) / I_lr

            theta_lr = obliquity(Lvec_lr)


            results['lr_mass'] = mtotal
            results['lr_mass_norm'] = 1.0
            results['slr_mass'] = 0
            results['slr_mass_norm'] = 0
            results['lr_core'] = Fe_lr
            results['lr_bulk_density'] = rho_lr / gcm3_to_ee3
            results['lr_radius'] = R_lr
            results['lr_Lvec'] = Lvec_lr
            results['lr_theta'] = theta_lr
            results['lr_angular_momentum'] = Lmag_lr * 2.4295e38  # J s
            results['lr_omega'] = omega_lr
            results['lr_omega_norm'] = omega_lr / omega_crit

            return results

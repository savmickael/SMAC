#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-
#=============================================================================================
# library for atmospheric correction using SMAC method (Rahman and Dedieu, 1994)
# Contains :
#      smac_inv : inverse smac model for atmospheric correction
#                          TOA==>Surface
#      smac dir : direct smac model
#                          Surface==>TOA
#      coefs : reads smac coefficients
#      PdeZ : #      PdeZ : Atmospheric pressure (in hpa) as a function of altitude (in meters)
 
# Written by O.Hagolle CNES, from the original SMAC C routine
#=============================================================================================
 
from math import *
import numpy as np
from numpy.polynomial.polynomial import polyval

from acquisition_conditions import AcquisitionConditions


def PdeZ(Z):
    """
    PdeZ : Atmospheric pressure (in hpa) as a function of altitude (in meters)
 
    """
    p = 1013.25 * pow(1 - 0.0065 * Z / 288.15, 5.31)
    return p


class SMACCoeff(object):
    def __init__(self, file_path):
        self._read(file_path)

    @staticmethod
    def _line_to_coef(line, nb_elt):
        temp = line.strip().split()
        if len(temp) == nb_elt:
            return tuple([float(elt) for elt in temp])
        else:
            raise ValueError

    def _read(self, file_path):
        with open(file_path) as fh:
            lines = fh.readlines()

        # H20
        self.ah2o, self.nh2o = SMACCoeff._line_to_coef(lines[0], 2)

        # O3
        self.ao3, self.no3 = SMACCoeff._line_to_coef(lines[1], 2)

        # O2
        self.ao2, self.no2, self.po2 = SMACCoeff._line_to_coef(lines[2], 3)

        # CO2
        self.aco2, self.nco2, self.pco2 = SMACCoeff._line_to_coef(lines[3], 3)

        # NH4
        self.ach4, self.nch4, self.pch4 = SMACCoeff._line_to_coef(lines[4], 3)

        # NO2
        self.ano2, self.nno2, self.pno2 = SMACCoeff._line_to_coef(lines[5], 3)

        # NO2
        self.aco, self.nco, self.pco = SMACCoeff._line_to_coef(lines[6], 3)

        # rayleigh and aerosol scattering
        self.a0s, self.a1s, self.a2s, self.a3s = SMACCoeff._line_to_coef(lines[7], 4)

        self.a0T, self.a1T, self.a2T, self.a3T = SMACCoeff._line_to_coef(lines[8], 4)

        self.taur, self.sr = SMACCoeff._line_to_coef(lines[9], 2)

        self.a0taup, self.a1taup = SMACCoeff._line_to_coef(lines[10], 2)

        self.wo, self.gc, = SMACCoeff._line_to_coef(lines[11], 2)

        self.a0P, self.a1P, self.a2P = SMACCoeff._line_to_coef(lines[12], 3)
        self.a3P, self.a4P = SMACCoeff._line_to_coef(lines[13], 2)

        self.Rest1, self.Rest2 = SMACCoeff._line_to_coef(lines[14], 2)
        self.Rest3, self.Rest4 = SMACCoeff._line_to_coef(lines[15], 2)

        self.Resr1, self.Resr2, self.Resr3 = SMACCoeff._line_to_coef(lines[16], 3)

        self.Resa1, self.Resa2 = SMACCoeff._line_to_coef(lines[17], 2)
        self.Resa3, self.Resa4 = SMACCoeff._line_to_coef(lines[18], 2)


class SMAC(object):
    def __init__(self, acq_conditions, pressure, taup550, uo3, uh2o, coef):
        self._acq_cond = acq_conditions
        self._pressure = pressure
        self._taup550 = taup550
        self._uo3 = uo3
        self._uh2o = uh2o
        self._coef = coef

        self._init_model()

    def _init_model(self):
        wo = self._coef.wo
        gc = self._coef.gc

        # calcul de la reflectance de surface  smac

        Peq = self._pressure / 1013.25

        # 2) aerosol optical depth in the spectral band, taup
        taup = self._coef.a0taup + self._coef.a1taup * self._taup550

        # 3) gaseous transmissions (downward and upward paths)
        u_atmo_composant = np.insert(np.power(Peq, [self._coef.po2, self._coef.pco2, self._coef.pch4,
                                                    self._coef.pno2, self._coef.pco]),
                                     0, [self._uo3, self._uh2o])

        # 4) if uh2o <= 0 and uo3<= 0 no gaseous absorption is computed
        t_atmo_composant = np.exp([self._coef.ao3, self._coef.ah2o, self._coef.ao2, self._coef.aco2, self._coef.ach4,
                                   self._coef.ano2, self._coef.aco] *
                                  np.power(u_atmo_composant * acq_cond.m,
                                           [self._coef.no3, self._coef.nh2o, self._coef.no2, self._coef.nco2,
                                            self._coef.nch4, self._coef.nno2, self._coef.nco]
                                           )
                                  )

        self._tg = np.prod(t_atmo_composant)

        # 5) Total scattering transmission
        self._ttetas = self._coef.a0T + self._coef.a1T * self._taup550 / acq_cond.us + \
                 (self._coef.a2T * Peq + self._coef.a3T) / (1. + self._acq_cond.us)  # downward
        self._ttetav = self._coef.a0T + self._coef.a1T * self._taup550 / acq_cond.uv + \
                 (self._coef.a2T * Peq + self._coef.a3T) / (1. + self._acq_cond.uv)  # upward

        # 6) spherical albedo of the atmosphere
        self._s = self._coef.a0s * Peq + polyval(self._taup550, [self._coef.a3s, self._coef.a1s, self._coef.a2s])

        # 9) rayleigh atmospheric reflectance
        ray_phase = 0.7190443 * (1. + (self._acq_cond.cksi * self._acq_cond.cksi)) + 0.0412742
        ray_ref = ((self._coef.taur * ray_phase) / (4 * self._acq_cond.usuv)) * Peq

        # 10) Residu Rayleigh
        Res_ray = polyval((self._coef.taur * ray_phase) / acq_cond.usuv,
                          [self._coef.Resr1, self._coef.Resr2, self._coef.Resr3])

        # 11) aerosol atmospheric reflectance
        aer_phase = polyval(self._acq_cond.ksid,
                            [self._coef.a0P, self._coef.a1P, self._coef.a2P, self._coef.a3P, self._coef.a4P])

        ak2 = (1. - wo) * (3. - wo * 3 * gc)
        ak = sqrt(ak2)
        e = -3 * acq_cond.us * acq_cond.us * wo / (4 * (1. - ak2 * acq_cond.us * acq_cond.us))
        f = -(1. - wo) * 3 * gc * acq_cond.us * acq_cond.us * wo / (4 * (1. - ak2 * acq_cond.us * acq_cond.us))
        dp = e / (3 * acq_cond.us) + acq_cond.us * f
        d = e + f
        b = 2 * ak / (3. - wo * 3 * gc)
        delta = np.exp(ak * taup) * (1. + b) * (1. + b) - np.exp(-ak * taup) * (1. - b) * (1. - b)
        ww = wo / 4.
        ss = acq_cond.us / (1. - ak2 * acq_cond.us * acq_cond.us)
        q1 = 2. + 3 * acq_cond.us + (1. - wo) * 3 * gc * acq_cond.us * (1. + 2 * acq_cond.us)
        q2 = 2. - 3 * acq_cond.us - (1. - wo) * 3 * gc * acq_cond.us * (1. - 2 * acq_cond.us)
        q3 = q2 * np.exp(-taup / acq_cond.us)
        c1 = ((ww * ss) / delta) * (q1 * np.exp(ak * taup) * (1. + b) + q3 * (1. - b))
        c2 = -((ww * ss) / delta) * (q1 * np.exp(-ak * taup) * (1. - b) + q3 * (1. + b))
        cp1 = c1 * ak / (3. - wo * 3 * gc)
        cp2 = -c2 * ak / (3. - wo * 3 * gc)
        z = d - wo * 3 * gc * acq_cond.uv * dp + wo * aer_phase / 4.
        x = c1 - wo * 3 * gc * acq_cond.uv * cp1
        y = c2 - wo * 3 * gc * acq_cond.uv * cp2
        aa1 = acq_cond.uv / (1. + ak * acq_cond.uv)
        aa2 = acq_cond.uv / (1. - ak * acq_cond.uv)
        aa3 = acq_cond.usuv / (acq_cond.us + acq_cond.uv)

        aer_ref = x * aa1 * (1. - np.exp(-taup / aa1))
        aer_ref = aer_ref + y * aa2 * (1. - np.exp(-taup / aa2))
        aer_ref = aer_ref + z * aa3 * (1. - np.exp(-taup / aa3))
        aer_ref = aer_ref / acq_cond.usuv

        # 12) Residu Aerosol
        Res_aer = polyval(taup * self._acq_cond.m * self._acq_cond.cksi,
                          [self._coef.Resa1, self._coef.Resa2, self._coef.Resa3, self._coef.Resa4])

        # 13)  Terme de couplage molecule / aerosol
        Res_6s = polyval((taup + self._coef.taur * Peq) * self._acq_cond.m * self._acq_cond.cksi,
                         [self._coef.Rest1, self._coef.Rest2, self._coef.Rest3, self._coef.Rest4])

        # 14) total atmospheric reflectance
        self._atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s

    def compute_toc(self, r_toa):
        # 15) Surface reflectance

        r_surf = r_toa - (self._atm_ref * self._tg)
        return r_surf / ((self._tg * self._ttetas * self._ttetav) + (r_surf * self._s))

    def compute_toa(self, r_surf):
        # 15) TOA reflectance

        return r_surf * self._tg * self._ttetas * self._ttetav / (1 - r_surf * self._s) + (self._atm_ref * self._tg)


if __name__ == "__main__":
    # Exemple
    theta_s = 45
    theta_v = 5
    phi_s = 200
    phi_v = -160
    r_toa = 0.2

    acq_cond = AcquisitionConditions(phi_s, theta_s, phi_v, theta_v)

    # lecture des coefs_smac
    nom_smac = 'COEFS/coef_FORMOSAT2_B1_CONT.dat'
    coefs = SMACCoeff(nom_smac)
    bd = 1
    r_surf = SMAC(acq_cond, 1013, 0.1, 0.3, 0.3, coefs).compute_toc(r_toa)
    r_toa = SMAC(acq_cond, 1013, 0.1, 0.3, 0.3, coefs).compute_toa(r_surf)

    print(r_toa, r_surf)

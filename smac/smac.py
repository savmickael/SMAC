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


class coeff:
    def __init__(self, smac_filename):
        with file(smac_filename) as f:
            lines = f.readlines()

        # H20
        temp = lines[0].strip().split()
        self.ah2o = float(temp[0])
        self.nh2o = float(temp[1])

        # O3
        temp = lines[1].strip().split()
        self.ao3 = float(temp[0])
        self.no3 = float(temp[1])

        # O2
        temp = lines[2].strip().split()
        self.ao2 = float(temp[0])
        self.no2 = float(temp[1])
        self.po2 = float(temp[2])

        # CO2
        temp = lines[3].strip().split()
        self.aco2 = float(temp[0])
        self.nco2 = float(temp[1])
        self.pco2 = float(temp[2])

        # NH4
        temp = lines[4].strip().split()
        self.ach4 = float(temp[0])
        self.nch4 = float(temp[1])
        self.pch4 = float(temp[2])

        # NO2
        temp = lines[5].strip().split()
        self.ano2 = float(temp[0])
        self.nno2 = float(temp[1])
        self.pno2 = float(temp[2])

        # NO2
        temp = lines[6].strip().split()
        self.aco = float(temp[0])
        self.nco = float(temp[1])
        self.pco = float(temp[2])

        # rayleigh and aerosol scattering
        temp = lines[7].strip().split()
        self.a0s = float(temp[0])
        self.a1s = float(temp[1])
        self.a2s = float(temp[2])
        self.a3s = float(temp[3])
        temp = lines[8].strip().split()
        self.a0T = float(temp[0])
        self.a1T = float(temp[1])
        self.a2T = float(temp[2])
        self.a3T = float(temp[3])
        temp = lines[9].strip().split()
        self.taur = float(temp[0])
        self.sr = float(temp[0])
        temp = lines[10].strip().split()
        self.a0taup = float(temp[0])
        self.a1taup = float(temp[1])
        temp = lines[11].strip().split()
        self.wo = float(temp[0])
        self.gc = float(temp[1])
        temp = lines[12].strip().split()
        self.a0P = float(temp[0])
        self.a1P = float(temp[1])
        self.a2P = float(temp[2])
        temp = lines[13].strip().split()
        self.a3P = float(temp[0])
        self.a4P = float(temp[1])
        temp = lines[14].strip().split()
        self.Rest1 = float(temp[0])
        self.Rest2 = float(temp[1])
        temp = lines[15].strip().split()
        self.Rest3 = float(temp[0])
        self.Rest4 = float(temp[1])
        temp = lines[16].strip().split()
        self.Resr1 = float(temp[0])
        self.Resr2 = float(temp[1])
        self.Resr3 = float(temp[2])
        temp = lines[17].strip().split()
        self.Resa1 = float(temp[0])
        self.Resa2 = float(temp[1])
        temp = lines[18].strip().split()
        self.Resa3 = float(temp[0])
        self.Resa4 = float(temp[1])


def smac_inv(r_toa, acq_cond, pressure, taup550, uo3, uh2o, coef):
    """
    r_surf=smac_inv( r_toa, tetas, phis, tetav, phiv,pressure,taup550, uo3, uh2o, coef)
    Corrections atmosphériques
    """
    ah2o = coef.ah2o
    nh2o = coef.nh2o
    ao3 = coef.ao3
    no3 = coef.no3
    ao2 = coef.ao2
    no2 = coef.no2
    po2 = coef.po2
    aco2 = coef.aco2
    nco2 = coef.nco2
    pco2 = coef.pco2
    ach4 = coef.ach4
    nch4 = coef.nch4
    pch4 = coef.pch4
    ano2 = coef.ano2
    nno2 = coef.nno2
    pno2 = coef.pno2
    aco = coef.aco
    nco = coef.nco
    pco = coef.pco
    a0s = coef.a0s
    a1s = coef.a1s
    a2s = coef.a2s
    a3s = coef.a3s
    a0T = coef.a0T
    a1T = coef.a1T
    a2T = coef.a2T
    a3T = coef.a3T
    taur = coef.taur
    sr = coef.sr
    a0taup = coef.a0taup
    a1taup = coef.a1taup
    wo = coef.wo
    gc = coef.gc
    a0P = coef.a0P
    a1P = coef.a1P
    a2P = coef.a2P
    a3P = coef.a3P
    a4P = coef.a4P
    Rest1 = coef.Rest1
    Rest2 = coef.Rest2
    Rest3 = coef.Rest3
    Rest4 = coef.Rest4
    Resr1 = coef.Resr1
    Resr2 = coef.Resr2
    Resr3 = coef.Resr3
    Resa1 = coef.Resa1
    Resa2 = coef.Resa2
    Resa3 = coef.Resa3
    Resa4 = coef.Resa4

    # calcul de la reflectance de surface  smac

    Peq = pressure/1013.25

    # 2) aerosol optical depth in the spectral band, taup
    taup = a0taup + a1taup * taup550

    # 3) gaseous transmissions (downward and upward paths)
    to3 = 1.
    th2o = 1.
    to2 = 1.
    tco2 = 1.
    tch4 = 1.

    uo2 = (Peq ** po2)
    uco2 = (Peq ** pco2)
    uch4 = (Peq ** pch4)
    uno2 = (Peq ** pno2)
    uco = (Peq ** pco)

    # 4) if uh2o <= 0 and uo3 <=0 no gaseous absorption is computed
    to3 = exp(ao3 * ((uo3 * acq_cond.m) ** no3))
    th2o = exp(ah2o * ((uh2o * acq_cond.m) ** nh2o))
    to2 = exp(ao2 * ((uo2 * acq_cond.m) ** no2))
    tco2 = exp(aco2 * ((uco2 * acq_cond.m) ** nco2))
    tch4 = exp(ach4 * ((uch4 * acq_cond.m) ** nch4))
    tno2 = exp(ano2 * ((uno2 * acq_cond.m) ** nno2))
    tco = exp(aco * ((uco * acq_cond.m) ** nco))
    tg = th2o * to3 * to2 * tco2 * tch4 * tco * tno2

    # 5) Total scattering transmission
    ttetas = a0T + a1T * taup550 / acq_cond.us + (a2T * Peq + a3T) / (1. + acq_cond.us)  # downward
    ttetav = a0T + a1T * taup550 / acq_cond.uv + (a2T * Peq + a3T) / (1. + acq_cond.uv)  # upward

    # 6) spherical albedo of the atmosphereS
    s = a0s * Peq + a3s + a1s * taup550 + a2s * taup550 ** 2

    # 9) rayleigh atmospheric reflectance
    ray_phase = 0.7190443 * (1. + (acq_cond.cksi*acq_cond.cksi)) + 0.0412742
    ray_ref = ((taur*ray_phase) / (4*acq_cond.usuv)) * Peq


    # 10) Residu Rayleigh
    Res_ray = polyval((taur*ray_phase) / acq_cond.usuv, [Resr1, Resr2, Resr3])

    # 11) aerosol atmospheric reflectance
    aer_phase = polyval(acq_cond.ksid, [a0P, a1P, a2P, a3P, a4P])

    ak2 = (1. - wo)*(3. - wo*3*gc)
    ak = sqrt(ak2)
    e = -3*acq_cond.us*acq_cond.us*wo / (4*(1. - ak2*acq_cond.us*acq_cond.us))
    f = -(1. - wo)*3*gc*acq_cond.us*acq_cond.us*wo / (4*(1. - ak2*acq_cond.us*acq_cond.us))
    dp = e / (3*acq_cond.us) + acq_cond.us*f
    d = e + f
    b = 2*ak / (3. - wo*3*gc)
    delta = np.exp(ak*taup)*(1. + b)*(1. + b) - np.exp(-ak*taup)*(1. - b)*(1. - b)
    ww = wo/4.
    ss = acq_cond.us / (1. - ak2*acq_cond.us*acq_cond.us)
    q1 = 2. + 3*acq_cond.us + (1. - wo)*3*gc*acq_cond.us*(1. + 2*acq_cond.us)
    q2 = 2. - 3*acq_cond.us - (1. - wo)*3*gc*acq_cond.us*(1. - 2*acq_cond.us)
    q3 = q2*np.exp(-taup/acq_cond.us)
    c1 = ((ww*ss) / delta) * (q1*np.exp(ak*taup)*(1. + b) + q3*(1. - b))
    c2 = -((ww*ss) / delta) * (q1*np.exp(-ak*taup)*(1. - b) + q3*(1. + b))
    cp1 = c1*ak / (3. - wo*3*gc)
    cp2 = -c2*ak / (3. - wo*3*gc)
    z = d - wo*3*gc*acq_cond.uv*dp + wo*aer_phase/4.
    x = c1 - wo*3*gc*acq_cond.uv*cp1
    y = c2 - wo*3*gc*acq_cond.uv*cp2
    aa1 = acq_cond.uv / (1. + ak*acq_cond.uv)
    aa2 = acq_cond.uv / (1. - ak*acq_cond.uv)
    aa3 = acq_cond.usuv / (acq_cond.us + acq_cond.uv)

    aer_ref = x*aa1*(1. - np.exp(-taup/aa1))
    aer_ref = aer_ref + y*aa2*(1. - np.exp(-taup / aa2))
    aer_ref = aer_ref + z*aa3*(1. - np.exp(-taup / aa3))
    aer_ref = aer_ref / acq_cond.usuv

    # 12) Residu Aerosol
    Res_aer = polyval(taup*acq_cond.m*acq_cond.cksi, [Resa1, Resa2, Resa3, Resa4])

    #  13)  Terme de couplage molecule / aerosol
    Res_6s = polyval((taup + taur*Peq)*acq_cond.m*acq_cond.cksi, [Rest1, Rest2, Rest3, Rest4])

    #  14) total atmospheric reflectance
    atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s

    # 15) Surface reflectance

    r_surf = r_toa - (atm_ref * tg)
    r_surf = r_surf / ((tg * ttetas * ttetav) + (r_surf * s))
 
    return r_surf


def smac_dir(r_surf, acq_cond, pressure, taup550, uo3, uh2o, coef):
    """
    r_toa=smac_dir ( r_surf, tetas, phis, tetav, phiv,pressure,taup550, uo3, uh2o, coef)
    Application des effets atmosphériques
    """
 
    ah2o = coef.ah2o
    nh2o = coef.nh2o
    ao3 = coef.ao3
    no3 = coef.no3
    ao2 = coef.ao2
    no2 = coef.no2
    po2 = coef.po2
    aco2 = coef.aco2
    nco2 = coef.nco2
    pco2 = coef.pco2
    ach4 = coef.ach4
    nch4 = coef.nch4
    pch4 = coef.pch4
    ano2 = coef.ano2
    nno2 = coef.nno2
    pno2 = coef.pno2
    aco = coef.aco
    nco = coef.nco
    pco = coef.pco
    a0s = coef.a0s
    a1s = coef.a1s
    a2s = coef.a2s
    a3s = coef.a3s
    a0T = coef.a0T
    a1T = coef.a1T
    a2T = coef.a2T
    a3T = coef.a3T
    taur = coef.taur
    sr = coef.sr
    a0taup = coef.a0taup
    a1taup = coef.a1taup
    wo = coef.wo
    gc = coef.gc
    a0P = coef.a0P
    a1P = coef.a1P
    a2P = coef.a2P
    a3P = coef.a3P
    a4P = coef.a4P
    Rest1 = coef.Rest1
    Rest2 = coef.Rest2
    Rest3 = coef.Rest3
    Rest4 = coef.Rest4
    Resr1 = coef.Resr1
    Resr2 = coef.Resr2
    Resr3 = coef.Resr3
    Resa1 = coef.Resa1
    Resa2 = coef.Resa2
    Resa3 = coef.Resa3
    Resa4 = coef.Resa4


    # calcul de la reflectance de surface  smac

    Peq = pressure/1013.25

    # 2) aerosol optical depth in the spectral band, taup
    taup = a0taup + a1taup * taup550

    # 3) gaseous transmissions (downward and upward paths)
    to3 = 1.
    th2o = 1.
    to2 = 1.
    tco2 = 1.
    tch4 = 1.

    uo2 = (Peq ** po2)
    uco2 = (Peq ** pco2)
    uch4 = (Peq ** pch4)
    uno2 = (Peq ** pno2)
    uco = (Peq ** pco)

    # 4) if uh2o <= 0 and uo3<= 0 no gaseous absorption is computed
    to3 = exp(ao3 * ((uo3*acq_cond.m) ** no3))
    th2o = exp(ah2o * ((uh2o*acq_cond.m) ** nh2o))
    to2 = exp(ao2 * ((uo2*acq_cond.m) ** no2))
    tco2 = exp(aco2 * ((uco2*acq_cond.m) ** nco2))
    tch4 = exp(ach4 * ((uch4*acq_cond.m) ** nch4))
    tno2 = exp(ano2 * ((uno2*acq_cond.m) ** nno2))
    tco = exp(aco * ((uco * acq_cond.m) ** nco))
    tg = th2o * to3 * to2 * tco2 * tch4 * tco * tno2

    # 5) Total scattering transmission
    ttetas = a0T + a1T*taup550/acq_cond.us + (a2T*Peq + a3T)/(1.+acq_cond.us)  # downward
    ttetav = a0T + a1T*taup550/acq_cond.uv + (a2T*Peq + a3T)/(1.+acq_cond.uv)  # upward

    # 6) spherical albedo of the atmosphere
    s = a0s * Peq + polyval(taup550, [a3s, a1s, a2s])

    # 9) rayleigh atmospheric reflectance
    ray_phase = 0.7190443 * (1. + (acq_cond.cksi*acq_cond.cksi)) + 0.0412742
    ray_ref = ((taur*ray_phase) / (4*acq_cond.usuv)) * Peq

    # 10) Residu Rayleigh
    Res_ray = polyval((taur*ray_phase) / acq_cond.usuv, [Resr1, Resr2, Resr3])

    # 11) aerosol atmospheric reflectance
    aer_phase = polyval(acq_cond.ksid, [a0P, a1P, a2P, a3P, a4P])

    ak2 = (1. - wo)*(3. - wo*3*gc)
    ak = sqrt(ak2)
    e = -3*acq_cond.us*acq_cond.us*wo / (4*(1. - ak2*acq_cond.us*acq_cond.us))
    f = -(1. - wo)*3*gc*acq_cond.us*acq_cond.us*wo / (4*(1. - ak2*acq_cond.us*acq_cond.us))
    dp = e / (3*acq_cond.us) + acq_cond.us*f
    d = e + f
    b = 2*ak / (3. - wo*3*gc)
    delta = np.exp(ak*taup)*(1. + b)*(1. + b) - np.exp(-ak*taup)*(1. - b)*(1. - b)
    ww = wo/4.
    ss = acq_cond.us / (1. - ak2*acq_cond.us*acq_cond.us)
    q1 = 2. + 3*acq_cond.us + (1. - wo)*3*gc*acq_cond.us*(1. + 2*acq_cond.us)
    q2 = 2. - 3*acq_cond.us - (1. - wo)*3*gc*acq_cond.us*(1. - 2*acq_cond.us)
    q3 = q2*np.exp(-taup/acq_cond.us)
    c1 = ((ww*ss) / delta) * (q1*np.exp(ak*taup)*(1. + b) + q3*(1. - b))
    c2 = -((ww*ss) / delta) * (q1*np.exp(-ak*taup)*(1. - b) + q3*(1. + b))
    cp1 = c1*ak / (3. - wo*3*gc)
    cp2 = -c2*ak / (3. - wo*3*gc)
    z = d - wo*3*gc*acq_cond.uv*dp + wo*aer_phase/4.
    x = c1 - wo*3*gc*acq_cond.uv*cp1
    y = c2 - wo*3*gc*acq_cond.uv*cp2
    aa1 = acq_cond.uv / (1. + ak*acq_cond.uv)
    aa2 = acq_cond.uv / (1. - ak*acq_cond.uv)
    aa3 = acq_cond.usuv / (acq_cond.us + acq_cond.uv)
 
    aer_ref = x*aa1 * (1. - np.exp(-taup/aa1))
    aer_ref = aer_ref + y*aa2*(1. - np.exp(-taup / aa2))
    aer_ref = aer_ref + z*aa3*(1. - np.exp(-taup / aa3))
    aer_ref = aer_ref / acq_cond.usuv
 
    # 12) Residu Aerosol
    Res_aer = polyval(taup * acq_cond.m * acq_cond.cksi, [Resa1, Resa2, Resa3, Resa4])
 
    # 13)  Terme de couplage molecule / aerosol
    Res_6s = polyval((taup + taur * Peq) * acq_cond.m * acq_cond.cksi,
                     [Rest1, Rest2, Rest3, Rest4])

    # 14) total atmospheric reflectance
    atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s
 
    # 15) TOA reflectance
 
    r_toa = r_surf*tg*ttetas*ttetav/(1-r_surf*s) + (atm_ref * tg)

    return r_toa


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
    coefs = coeff(nom_smac)
    bd = 1
    r_surf = smac_inv(r_toa, acq_cond, 1013, 0.1, 0.3, 0.3, coefs)
    r_toa2 = smac_dir(r_surf, acq_cond, 1013, 0.1, 0.3, 0.3, coefs)

    print r_toa, r_surf, r_toa2

# -*- coding: utf-8 -*-
"""
:author: Mickael Savinaud
:organization: CS SI
:copyright: CNES, Centre National d'Etudes SpatialesCS SI
:license: see LICENSE file
:created: 2019, Aug, 7
"""
from __future__ import absolute_import

from math import acos, degrees
from numpy import cos as np_cos , radians as np_radians, sqrt as np_sqrt, array as np_array, subtract as np_subtract


class AcquisitionConditions(object):

    def __init__(self, phi_s, theta_s, phi_v, theta_v):
        # solar angles
        self._phi_s = phi_s
        self._theta_s = theta_s
        # viewing angles
        self._phi_v = phi_v
        self._theta_v = theta_v

        self._us = np_cos(np_radians(self._theta_s))
        self._uv = np_cos(np_radians(self._theta_v))

        # air mass
        self._m = 1 / self._us + 1 / self._uv

        # scattering angle cosine
        self._cksi = - (self._us * self._uv +
                        np_sqrt(1. - self._us * self._us) *
                        np_sqrt(1. - self._uv * self._uv) *
                        np_cos(np_radians(np_subtract(self._phi_s, self._phi_v))))
        if self._cksi < -1:
            self._cksi = -1.0

        # scattering angle in degree
        self._ksiD = degrees(acos(self._cksi))

        self._usuv = self._us * self._uv

    @property
    def us(self):
        return self._us

    @property
    def uv(self):
        return self._uv

    @property
    def usuv(self):
        return self._usuv

    @property
    def m(self):
        """
        Air mass
        :return: air mass
        :rtype: float
        """
        return self._m

    @property
    def cksi(self):
        """
        Get scattering angle cosine
        :return: scattering angle cosine
        :rtype float
        """
        return self._cksi

    @property
    def ksid(self):
        """
        scattering angle in degree
        :return:
        """
        return self._ksiD


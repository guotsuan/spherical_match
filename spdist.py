#! /usr/bin/env python2
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 qguo <qguo@lupus.aip.de>
#
# Distributed under terms of the MIT license.
#
# Last modified: 2015 Apr 08

"""
Modules for tools to find the nearest neighbour or
all neighbours in within certain diameter angle

"""
from __future__ import division
import numpy as np
from scipy.spatial import cKDTree as KDT
from scipy.spatial.distance import cdist as cdist

def radec2xyz(ra, dec, degree=False):

    ra = np.asarray(ra)
    dec = np.asarray(dec)

    if degree:
        ra_r = np.radians(ra)
        dec_r = np.radians(dec)
    else:
        ra_r = ra
        dec_r = dec

    x = np.cos(ra_r) * np.cos(dec_r)
    y = np.sin(ra_r) * np.cos(dec_r)
    z = np.sin(dec_r)
    coords = np.empty((x.size, 3))
    coords[:, 0] = x
    coords[:, 1] = y
    coords[:, 2] = z

    return x, y, z


def build_sp_kdtree(ra, dec, degree=False):
    """
    ra and dec are in radian, if not set degree to True
    """

    x, y, z = radec2xyz(ra, dec, degree=degree)
    coords = np.empty((x.size, 3))
    coords[:, 0] = x
    coords[:, 1] = y
    coords[:, 2] = z

    tree = KDT(coords)

    return tree



def sph_query_ball(tree, ra, dec, r, degree=False):
    """
    Note: only query the the cubic distance, not diamter angles
          need to perform another selection
    """

    x, y, z = radec2xyz(ra, dec, degree=degree)

    coords = np.empty((x.size, 3))
    coords[:, 0] = x
    coords[:, 1] = y
    coords[:, 2] = z
    idx = tree.query_ball_point(coords, r)

    return idx

def sph_sp_xyz(x, y, z, ra2, dec2, degree=False):
    x1, y1, z1= x, y, z
    x2, y2, z2= radec2xyz(ra2, dec2, degree=degree)

    p1 = np.vstack((x1, y1, z1)).T
    p2 = np.vstack((x2, y2, z2)).T


    mod_p1 = np.linalg.norm(p1, axis=1)
    mod_p2 = np.linalg.norm(p2, axis=1)

    mod_p1p2= np.outer(mod_p1, mod_p2)


    dot = np.inner(p1, p2)/mod_p1p2

    angle = np.arccos(dot)

    return angle

def sph_sp(ra1, dec1, ra2, dec2, degree=False):
    x1, y1, z1= radec2xyz(ra1, dec1, degree=degree)
    x2, y2, z2= radec2xyz(ra2, dec2, degree=degree)

    p1 = np.vstack((x1, y1, z1)).T
    p2 = np.vstack((x2, y2, z2)).T


    mod_p1 = np.linalg.norm(p1, axis=1)
    mod_p2 = np.linalg.norm(p2, axis=1)

    mod_p1p2= np.outer(mod_p1, mod_p2)


    dot = np.inner(p1, p2)/mod_p1p2

    angle = np.arccos(dot)

    return angle

def cube_sp(ra1, dec1, ra2, dec2, degree=False):

    x1, y1, z1= radec2xyz(ra1, dec1, degree=degree)
    x2, y2, z2= radec2xyz(ra2, dec2, degree=degree)

    p1 = np.vstack((x1, y1, z1)).T
    p2 = np.vstack((x2, y2, z2)).T

    return cdist(p1, p2)

if __name__ == '__main__':


    ra = np.random.rand(10000000) * np.pi
    dec = np.random.rand(10000000) * np.pi - np.pi/2.0

    ra0, dec0 = [0, 0.1, 0.2], [0, 0.1, 0.2]

    #tree = build_sp_kdtree(ra, dec)

    #idx = sph_query_ball(tree, ra0,  dec0, 1.2)

    #for i, idx_now in enumerate(idx):
        #all_angle = sph_sp(ra0[i], dec0[i], ra[idx_now], dec[idx_now])

        #idx = all_angle < 1.2
        #print all_angle[idx].size, 'kdetree: result'


    for i, _ in enumerate(ra0):
        angles_b = sph_sp(ra0[i], dec0[i], ra, dec)

        idx = angles_b < 1.2

        print angles_b[idx]. size, 'brutal: result'





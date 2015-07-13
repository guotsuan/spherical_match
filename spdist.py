#! /usr/bin/env python2
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 qguo <qguo@lupus.aip.de>
#
# Distributed under terms of the MIT license.
#
# Last modified: 2015 Jul 13

"""
Modules for tools to find the nearest neighbour or
all neighbours in within certain diameter angle

"""
from __future__ import division
import numpy as np
from scipy.spatial import cKDTree as KDT
from scipy.spatial.distance import cdist as cdist


class gals_tree:

    def __init__(self, ra_all, dec_all, degree=False):
        self.ra_all = ra_all
        self.dec_all = dec_all
        self.tree = build_sp_kdtree(self.ra_all, self.dec_all, degree=degree)


    def sph_query_ball_asp(self, ra_cen, dec_cen, r, degree=False):
        """
        Note: r: diamter angles seperation, not 3D distants
        """

        idx = sph_query_ball(self.tree, ra_cen, dec_cen, r, degree=degree)

        idx_result = []
        if np.array(r).size == 1:
            r_arr = np.zeros_like(ra_cen) + r
        else:
            r_arr = r

        for idx_now, ra_now, dec_now, rnow in zip(idx, ra_cen, dec_cen,
                                            r_arr):
            angle_sp = sph_sp(self.ra_all[idx_now], self.dec_all[idx_now],
                                ra_now, dec_now, degree=degree)
            good_idx = np.squeeze(angle_sp) < rnow

            idx_this = np.array(idx_now)[np.atleast_1d(good_idx)]
            idx_result.append(idx_this)


        return idx_result


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


def sph_query_ball_asp(ra_all, dec_all, ra_cen, dec_cen, r, degree=False):
    """
    Note: r: diamter angles seperation, not 3D distants
    """

    tree = build_sp_kdtree(ra_all, dec_all, degree=degree)

    idx = sph_query_ball(tree, ra_cen, dec_cen, r, degree=degree)

    idx_result = []
    if np.array(r).size == 1:
        r_arr = np.zeros_like(ra_cen) + r
    else:
        r_arr = r

    for idx_now, ra_now, dec_now, rnow in zip(idx, ra_cen, dec_cen,
                                           r_arr):
        angle_sp = sph_sp(ra_all[idx_now], dec_all[idx_now],
                            ra_now, dec_now, degree=degree)
        good_idx = np.squeeze(angle_sp) < rnow

        idx_this = np.array(idx_now)[np.atleast_1d(good_idx)]
        idx_result.append(idx_this)


    return idx_result

def sph_query_ball_asp_with_tree(tree, ra_cen, dec_cen, r, degree=False):
    """
    Note: r: diamter angles seperation, not 3D distants
    """

    idx = sph_query_ball(tree, ra_cen, dec_cen, r, degree=degree)

    idx_result = []
    if np.array(r).size == 1:
        r_arr = np.zeros_like(ra_cen) + r
    else:
        r_arr = r

    for idx_now, ra_now, dec_now, rnow in zip(idx, ra_cen, dec_cen,
                                           r_arr):
        angle_sp = sph_sp(ra_all[idx_now], dec_all[idx_now],
                            ra_now, dec_now, degree=degree)
        good_idx = np.squeeze(angle_sp) < rnow

        idx_this = np.array(idx_now)[np.atleast_1d(good_idx)]
        idx_result.append(idx_this)


    return idx_result


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

    if np.array(r).size == 1:
        idx = tree.query_ball_point(coords, r)
    elif x.shape == np.array(r).shape:
        idx = []
        for pnow, rnow in zip(coords, r):
            idx_now = tree.query_ball_point(pnow, rnow)
            idx.append(idx_now)

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

    dot = np.clip(dot, -1.0, 1.0)

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

    dot = np.clip(dot, -1.0, 1.0)

    angle = np.arccos(dot)

    return angle


def sph_sp_d(ra1, dec1, ra2, dec2, degree=False):
    #x1, y1, z1= radec2xyz(ra1, dec1, degree=degree)
    #x2, y2, z2= radec2xyz(ra2, dec2, degree=degree)

    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)

    cosangle = np.sin(dec1 + 0.5*np.pi) * np.sin(dec2 + 0.5*np.pi) * \
        np.cos(ra1 - ra2) + np.cos(dec1 + 0.5*np.pi) * \
        np.cos(dec2 + 0.5 *np.pi)

    angle = np.arccos(cosangle)

    print cosangle

    return angle

def cube_sp(ra1, dec1, ra2, dec2, degree=False):

    x1, y1, z1= radec2xyz(ra1, dec1, degree=degree)
    x2, y2, z2= radec2xyz(ra2, dec2, degree=degree)

    p1 = np.vstack((x1, y1, z1)).T
    p2 = np.vstack((x2, y2, z2)).T

    return cdist(p1, p2)

if __name__ == '__main__':


    ra = np.random.rand(1000) * np.pi
    dec = np.random.rand(1000) * np.pi - np.pi/2.0

    #ra0, dec0 = [0, 0.1, 0.2], [0, 0.1, 0.2]
    ra0, dec0 = [0.1, 0.1, 0.1], [-1.3, -1.3, -1.3]

    tree = build_sp_kdtree(ra, dec)

    idx = sph_query_ball(tree, ra0,  dec0, [1.8, 1.9, 2.0])

    for i, idx_now in enumerate(idx):

        print len(idx_now), 'kdetree: result, 3D distant'

        #print ra0, dec0
        #angle = sph_sp(ra[idx_now], dec[idx_now], ra0, dec0)
        #print angle

    idx_sp = sph_query_ball_asp(ra, dec, ra0, dec0, [1.8, 1.9, 2.0])


    for i, idx_now in enumerate(idx_sp):

        print len(idx_now), 'kdetree: result, angular SP'


    for ra_now, dec_now, r in zip(ra0, dec0, [1.8, 1.9, 2.0]):
        spnow = sph_sp(ra, dec, ra_now, dec_now)

        idx = spnow < r
        print ra[np.squeeze(idx)].shape, 'brutal resutls: angular SP'









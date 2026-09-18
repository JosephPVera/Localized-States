#!/usr/bin/env python3
# 2025-03

"""
Usage:
     python3 ks-orbital.py [-s] [-k] [-b] [-n] [-ng]
     
Plot the real and imaginary part of the selected KS orbital following their spin, kpoint, and band.
Based on: https://github.com/QijingZheng/VaspBandUnfolding/tree/master

OUTPUTS: wfc_r.vasp and wfc_i.vasp
"""
# .wfc_r is same to .get_ps_wfc
# usage: ks-orbital.py -s -k -b 

import argparse
from vaspwfc import vaspwfc

parser = argparse.ArgumentParser(description="Extract and save KS orbital from WAVECAR.")

parser.add_argument("-s", "--ispin", type=int, required=True, help="Spin index")
parser.add_argument("-k", "--ikpt", type=int, required=True, help="K-point index")
parser.add_argument("-b", "--iband", type=int, required=True, help="Band index")
parser.add_argument("-n", "--ngrid_mult", type=int, default=4, help="Grid multiplier (optional)")
parser.add_argument("-ng", "--ngamma", action="store_true", help="For no Gamma-only WAVECAR, ie, several kpoints")

args = parser.parse_args()

if args.ngamma:
    wav = vaspwfc('WAVECAR')
else:
    wav = vaspwfc('WAVECAR', lgamma=True) # , gamma_half='x')

# KS orbital in real space, double the size of the FT grid
#phi = wav.wfc_r(ikpt=1, iband=430, ngrid=wav._ngrid * 2)
phi = wav.get_ps_wfc(ispin=args.ispin, ikpt=args.ikpt, iband=args.iband, ngrid=wav._ngrid * args.ngrid_mult)

# Save the orbital into files. Since the wavefunction consist of complex numbers, the real and imaginary part are saved separately.
wav.save2vesta(phi, prefix=f'wfc_s{args.ispin}_k{args.ikpt}_b{args.iband}')  #, poscar='POSCAR')

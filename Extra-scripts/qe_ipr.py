#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-09

"""
Usage:
      python3 qe_ipr.py
      
Reads Quantum ESPRESSO binary wavefunction files (Fortran-unformatted,) 
for a spin-polarized Gamma-point calculation and computes the Inverse 
Participation Ratio (IPR) of every band, for both spin channels, in a 
single pass.

Read:
- tmp/*.save/{wfcup1.dat, wfcdw1.dat}
- Energies from nscf/*.out

Format (from Modules/io_base.f90, subroutine read_wfc, QE master branch):

    record 1: ik, xk(3), ispin, gamma_only, scalef
    record 2: ngw, igwx, npol, nbnd
    record 3: b1(3), b2(3), b3(3)
    record 4: mill(3, igwx)                 <- Miller indices of each G
    record 5..(4+nbnd): wtmp(npol*igwx)     <- one complex band per record

ONLY VALID FOR A GAMMA-POINT FILE (xk = 0): at Gamma, psi(r) = sum_G
c_G exp(i G.r), with NO k-dependent phase factor, so a plain inverse
FFT of the (mill, c_G) pairs onto the real-space FFT grid reconstructs
psi(r) directly. For any other k-point this script would need the
e^{ik.r} factor and is NOT applicable as-is.

If the SCF run used 'K_POINTS gamma' (the standard Gamma-only trick),
gamma_only=True is stored in the file and only ~half of the G-sphere
is written (say G_z >= 0, plus special handling at G=0); the missing
half is reconstructed here via c(-G) = conj(c(G)), which is exactly
the condition satisfied by a real-space charge-density-generating
wavefunction. If gamma_only=False (a general k-point run, even though
this particular k happens to be 0,0,0), all G's are already stored
explicitly and no mirroring is done.

IPR is SCALE-INVARIANT (IPR = sum(rho^2)/sum(rho)^2), so the overall
FFT normalization convention and QE's 'scalef' never matter here, only 
the RELATIVE distribution of |psi(r)|^2 across the grid does.
"""

import glob
import os
import re
import sys

import numpy as np

TMP_DIR = "../tmp"
NSCF_DIR = "../nscf"
NR = (90, 90, 90)
OUT_PATH = "ipr.dat"

# energy parsing 
FLOAT_RE = re.compile(r"[-+]?\d+\.\d+")
SPIN_UP_RE = re.compile(r"spin\s+up", re.IGNORECASE)
SPIN_DOWN_RE = re.compile(r"spin\s+down", re.IGNORECASE)
KPOINT_BANDS_RE = re.compile(
    r"k\s*=\s*[-+0-9.\s]+\([^)]*\)\s*bands\s*\(ev\):", re.IGNORECASE
)

KPOINT_VALUE_RE = re.compile(
    r"k\s*=\s*([-+0-9.\s]+?)\s*\([^)]*\)\s*bands\s*\(ev\):", re.IGNORECASE
)

class FortranUnformattedReader:
    """Minimal sequential reader for gfortran/ifort-style Fortran
    unformatted files: each record is wrapped in a 4-byte (default) or
    8-byte leading/trailing byte count. Falls back from 4- to 8-byte
    markers automatically if the first record's declared length does
    not match what read_wfc's own header says it should be (see
    read_qe_wfc_header), so a non-default compiler build is still
    handled rather than silently misread."""

    def __init__(self, path, marker_bytes=4):
        self.f = open(path, 'rb')
        self.marker_bytes = marker_bytes
        self.marker_dtype = {4: '<u4', 8: '<u8'}[marker_bytes]

    def read_record(self):
        head = self.f.read(self.marker_bytes)
        if len(head) < self.marker_bytes:
            raise EOFError("end of file")
        n = np.frombuffer(head, dtype=self.marker_dtype)[0]
        data = self.f.read(int(n))
        tail = self.f.read(self.marker_bytes)
        n2 = np.frombuffer(tail, dtype=self.marker_dtype)[0]
        if n2 != n:
            raise RuntimeError(f"record marker mismatch ({n} vs {n2}), "
                                f"wrong marker byte width ({self.marker_bytes}) "
                                f"for this file?")
        return data

    def close(self):
        self.f.close()

def _open_with_marker_autodetect(path):
    """Tries 4-byte record markers (the near-universal default for
    gfortran/ifort); if the very first record's byte count doesn't
    match record 1's known fixed size, retries with 8-byte markers
    before giving up. Record 1 size = 4(ik) + 24(xk) + 4(ispin) +
    4(gamma_only) + 8(scalef) = 44 bytes (standard 4-byte Fortran
    INTEGER/LOGICAL, 8-byte DOUBLE PRECISION, QE's normal build)."""
    expected_rec1 = 4 + 8 * 3 + 4 + 4 + 8
    for mb in (4, 8):
        r = FortranUnformattedReader(path, marker_bytes=mb)
        head = r.f.read(mb)
        n = np.frombuffer(head, dtype=r.marker_dtype)[0]
        r.f.seek(0)
        if n == expected_rec1:
            return r
        r.close()
    raise RuntimeError(f"could not determine record-marker width for {path} "
                        f"(tried 4 and 8 bytes; neither matches the expected "
                        f"{expected_rec1}-byte first record). This file may "
                        f"not be a QE wfc*.dat file, or uses an unusual build.")

def find_save_dir(tmp_dir):
    """Looks inside tmp_dir for exactly one '*.save' directory,
    regardless of its prefix name, and returns its full path. Raises
    if none or more than one is found (in which case the caller needs
    to disambiguate manually)."""
    if not os.path.isdir(tmp_dir):
        raise FileNotFoundError(f"'{tmp_dir}' does not exist or is not a directory")
    candidates = sorted(
        p for p in glob.glob(os.path.join(tmp_dir, "*.save")) if os.path.isdir(p)
    )
    if not candidates:
        raise FileNotFoundError(f"no '*.save' directory found inside '{tmp_dir}'")
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple '*.save' directories found inside '{tmp_dir}': "
            f"{candidates}, expected only one."
        )
    return candidates[0]

def find_out_file(directory):
    """Looks inside directory for a single '*.out' file, ignoring
    Slurm's own 'slurm-<jobid>.out' logs, the same idea as
    find_save_dir, applied to the nscf output instead of the .save
    folder. Warns and uses the first (sorted) match if more than one
    is found."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"'{directory}' does not exist or is not a directory")
    all_out = glob.glob(os.path.join(directory, "*.out"))
    out_files = sorted(
        f for f in all_out if not re.match(r"^slurm-\d+\.out$", os.path.basename(f))
    )
    if not out_files:
        raise FileNotFoundError(
            f"no '*.out' file found inside '{directory}' (ignoring slurm-*.out)"
        )
    if len(out_files) > 1:
        print(f"Warning: multiple '.out' files found in '{directory}': "
              f"{out_files}, using: {out_files[0]}", file=sys.stderr)
    return out_files[0]

def split_spin_sections(text):
    """Splits a QE .out file into its SPIN UP / SPIN DOWN halves. If no spin
    markers are found (a non-spin-polarized run), everything is treated as 
    the single 'UP' section and 'DOWN' is empty."""
    up_m = SPIN_UP_RE.search(text)
    down_m = SPIN_DOWN_RE.search(text)
    if up_m and down_m:
        return {"UP": text[up_m.end():down_m.start()], "DOWN": text[down_m.end():]}
    return {"UP": text, "DOWN": ""}

def extract_band_energies(section_text):
    """Finds the FIRST 'k = ... bands (ev):' eigenvalue block in
    section_text. This script assumes a single, Gamma, k-point,
    matching the wfc*.dat files used for the IPR calculation and
    returns its band energies (eV) in band order (QE always lists
    them in increasing band index, wrapped across several lines, with
    a blank line marking the end of the block)."""
    m = KPOINT_BANDS_RE.search(section_text)
    if not m:
        return []
    block_start = m.end()
    # QE always inserts one blank line right after "bands (ev):" before the
    # actual numbers start -- skip past it (and any other leading
    # whitespace) so the blank-line search below doesn't mistake it for the
    # end of the block.
    after_header = section_text[block_start:]
    lead_ws = len(after_header) - len(after_header.lstrip("\n"))
    block_start += lead_ws
    blank = re.search(r"\n\s*\n", section_text[block_start:])
    block_end = block_start + blank.start() if blank else len(section_text)
    return [float(x) for x in FLOAT_RE.findall(section_text[block_start:block_end])]

def extract_kpoint(section_text):
    """Finds the FIRST 'k = ... bands (ev):' header line in section_text
    (same match as extract_band_energies) and returns its k-point
    coordinates as a tuple of 3 floats, e.g. (0.0, 0.0, 0.0). Returns
    None if no such header is found."""
    m = KPOINT_VALUE_RE.search(section_text)
    if not m:
        return None
    coords = [float(x) for x in FLOAT_RE.findall(m.group(1))]
    if len(coords) != 3:
        return None
    return tuple(coords)

def read_qe_wfc_header(path):
    """Reads records 1-4 (everything except the band coefficients) and
    returns a dict with the header info plus an open reader positioned
    right after record 4, ready to read band records 5, 6, ... in
    order via reader.read_record()."""
    r = _open_with_marker_autodetect(path)

    rec1 = r.read_record()
    ik = np.frombuffer(rec1, dtype='<i4', count=1, offset=0)[0]
    xk = np.frombuffer(rec1, dtype='<f8', count=3, offset=4)
    ispin = np.frombuffer(rec1, dtype='<i4', count=1, offset=4 + 24)[0]
    gamma_flag = np.frombuffer(rec1, dtype='<i4', count=1, offset=4 + 24 + 4)[0]
    scalef = np.frombuffer(rec1, dtype='<f8', count=1, offset=4 + 24 + 4 + 4)[0]
    gamma_only = bool(gamma_flag)

    rec2 = r.read_record()
    ngw, igwx, npol, nbnd = np.frombuffer(rec2, dtype='<i4', count=4)

    rec3 = r.read_record()
    bvecs = np.frombuffer(rec3, dtype='<f8', count=9).reshape(3, 3)
    b1, b2, b3 = bvecs

    rec4 = r.read_record()
    mill = np.frombuffer(rec4, dtype='<i4', count=3 * int(igwx)).reshape(int(igwx), 3)

    return {
        'reader': r, 'ik': int(ik), 'xk': xk, 'ispin': int(ispin),
        'gamma_only': gamma_only, 'scalef': float(scalef),
        'ngw': int(ngw), 'igwx': int(igwx), 'npol': int(npol),
        'nbnd': int(nbnd), 'b1': b1, 'b2': b2, 'b3': b3, 'mill': mill,
    }

def band_density_from_coeffs(mill, coeffs, gamma_only, nr):
    """Places the (mill, coeffs) plane-wave expansion of ONE band onto
    an (nr1,nr2,nr3) FFT grid and inverse-FFTs it to real space,
    mirroring G -> -G with a conjugated coefficient first if
    gamma_only (see module docstring). Returns |psi(r)|^2 as a real
    (nr1,nr2,nr3) array. Valid only at k=0 (Gamma): no e^{ik.r} phase
    is applied."""
    nr1, nr2, nr3 = nr
    arr = np.zeros((nr1, nr2, nr3), dtype=complex)

    ix = mill[:, 0] % nr1
    iy = mill[:, 1] % nr2
    iz = mill[:, 2] % nr3
    arr[ix, iy, iz] = coeffs

    if gamma_only:
        is_origin = (mill[:, 0] == 0) & (mill[:, 1] == 0) & (mill[:, 2] == 0)
        neg = -mill[~is_origin]
        nix = neg[:, 0] % nr1
        niy = neg[:, 1] % nr2
        niz = neg[:, 2] % nr3
        arr[nix, niy, niz] = np.conj(coeffs[~is_origin])

    psi = np.fft.ifftn(arr)
    return np.abs(psi) ** 2

def ipr_from_density(rho):
    rho = rho.astype(float)
    denom = np.sum(rho)
    if denom == 0:
        raise RuntimeError("sum(rho) == 0 for this band, cannot compute IPR")
    return float(np.sum(rho ** 2) / denom ** 2)

def compute_channel_ipr(path, nr, label=""):
    """Reads one spin-channel wfc*.dat file sequentially (O(nbnd) time,
    ONE file open, no repeated pp.x process launches) and returns
    (bands, raw_ipr) lists, in band order, for every band in the
    file."""
    h = read_qe_wfc_header(path)
    r = h['reader']
    npol, igwx, nbnd = h['npol'], h['igwx'], h['nbnd']
    mill = h['mill']

    print(f"[{label}] {os.path.basename(path)}: ik={h['ik']}  xk={h['xk']}  "
          f"ispin={h['ispin']}  gamma_only={h['gamma_only']}  npol={h['npol']}  "
          f"ngw={h['ngw']}  igwx={h['igwx']}  nbnd={h['nbnd']}", file=sys.stderr)

    if abs(h['xk']).max() > 1e-6:
        print(f"WARNING [{label}]: xk = {h['xk']} is not (0,0,0). This file "
              f"is NOT a Gamma-point wavefunction, and this script's "
              f"real-space reconstruction (no e^{{ik.r}} phase) is NOT valid "
              f"for it.", file=sys.stderr)

    if npol != 1:
        raise NotImplementedError("npol=2 (noncollinear) wavefunctions are "
                                   "not handled by this script.")

    bands, raw = [], []
    for b in range(1, nbnd + 1):
        rec = r.read_record()
        coeffs = np.frombuffer(rec, dtype='<c16', count=igwx)
        rho = band_density_from_coeffs(mill, coeffs, h['gamma_only'], nr)
        val = ipr_from_density(rho)
        bands.append(b)
        raw.append(val)
        print(f"[{label}] band {b}: IPR = {val:.10e}", file=sys.stderr)
    r.close()
    return bands, raw

def minmax_normalize(values):
    """Min-max normalization to [0, 1]: (x - min) / (max - min), computed
    over the WHOLE given collection of values (this is why every band
    in a channel must be computed first, before any normalized value
    can be written for that channel, unlike the raw IPR, the
    normalized value of one band depends on every other band in its
    own channel). If all values are equal (max == min, a degenerate
    edge case), every normalized value is set to 0.0 with a warning,
    since (x-min)/(max-min) would otherwise divide by zero."""
    values = np.asarray(values, dtype=float)
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        print("WARNING: all IPR values are identical in this channel,"
              "min-max normalization is undefined; writing 0.0 for every "
              "band.", file=sys.stderr)
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)

def main():
    save_dir = find_save_dir(TMP_DIR)
    print(f"Using save directory: {save_dir}", file=sys.stderr)

    wfcup_path = os.path.join(save_dir, "wfcup1.dat")
    wfcdw_path = os.path.join(save_dir, "wfcdw1.dat")
    for p in (wfcup_path, wfcdw_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"expected file not found: {p}")

    bands_up, raw_up = compute_channel_ipr(wfcup_path, NR, label="up")
    bands_dw, raw_dw = compute_channel_ipr(wfcdw_path, NR, label="dw")

    if bands_up != bands_dw:
        raise RuntimeError(
            "up and dw channels do not have the same set of band indices "
            f"({len(bands_up)} vs {len(bands_dw)} bands), cannot write a "
            "single aligned ipr.dat. Check that wfcup1.dat and wfcdw1.dat "
            "come from the same calculation."
        )
    bands = bands_up

    norm_up = minmax_normalize(raw_up)
    norm_dw = minmax_normalize(raw_dw)

    nscf_out = find_out_file(NSCF_DIR)
    print(f"Using nscf output file: {nscf_out}", file=sys.stderr)
    with open(nscf_out, "r", errors="ignore") as f:
        out_text = f.read()
    spin_sections = split_spin_sections(out_text)
    energies_up = extract_band_energies(spin_sections["UP"])
    energies_dw = extract_band_energies(spin_sections["DOWN"])
    if not energies_dw:
        # non-spin-polarized run (or DOWN section absent): mirror UP so the
        # file still has 7 columns, with Energy(dw) == Energy(up).
        energies_dw = energies_up
    if len(energies_up) != len(bands) or len(energies_dw) != len(bands):
        raise RuntimeError(
            f"found {len(energies_up)} spin-up and {len(energies_dw)} "
            f"spin-down band energies in '{nscf_out}' but {len(bands)} "
            f"bands in the wfc files. check that the nscf run and the "
            f"wfc*.dat files correspond to the same calculation."
        )

    kpoint = extract_kpoint(spin_sections["UP"]) or extract_kpoint(spin_sections["DOWN"])

    with open(OUT_PATH, 'w') as out:
        if kpoint is not None:
            out.write(f"# k-point: {kpoint[0]:.6f}  {kpoint[1]:.6f}  {kpoint[2]:.6f}\n")
        else:
            print("WARNING: could not find a k-point value in the nscf "
                  "output, writing ipr.dat without a k-point line.",
                  file=sys.stderr)
        out.write("# band     ipr_up            ipr_norm_up       "
                   "ipr_dw            ipr_norm_dw       Energy_up(eV)   "
                   "Energy_dw(eV)\n")
        for b, u, nu, d, nd, eu, ed in zip(
                bands, raw_up, norm_up, raw_dw, norm_dw, energies_up, energies_dw):
            out.write(f"{b:6d}   {u:.10e}   {nu:.10e}   "
                       f"{d:.10e}   {nd:.10e}   {eu:12.6f}   {ed:12.6f}\n")

    print(f"\nWrote {len(bands)} bands to {OUT_PATH}", file=sys.stderr)
    print(f"  up: min IPR = {min(raw_up):.6e} at band "
          f"{bands[int(np.argmin(raw_up))]}, max IPR = {max(raw_up):.6e} at "
          f"band {bands[int(np.argmax(raw_up))]}", file=sys.stderr)
    print(f"  dw: min IPR = {min(raw_dw):.6e} at band "
          f"{bands[int(np.argmin(raw_dw))]}, max IPR = {max(raw_dw):.6e} at "
          f"band {bands[int(np.argmax(raw_dw))]}", file=sys.stderr)

if __name__ == "__main__":
    main()

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import Angle
from gammapy.maps import MapAxes, MapAxis
from gammapy.irf.psf.parametric import ParametricPSF #, PSFKing

__all__ = ["SwiftPSF"]

class SwiftPSF(ParametricPSF):
    """Swift XRT PSF depending on energy and offset.

    The PSF is described by a King + Gauss profile:
    PSF = P0 * Gauss(P1) + (1 - P0) * King(P2, P3)

    Parameters P0, P1, P2, P3 depend on energy and offset:
    P_i = COEF0 + COEF1 * offaxis + COEF2 * energy + COEF3 * energy * offaxis

    where offaxis is in arcmin/0.1 and energy in keV/0.1 respectively. 

    Units assumed for coefficients:
    - Energy: keV/0.1
    - Off-axis angle: arcmin/0.1
    - P1 (Gauss sigma): pixels (1 pixel = 2.357 arcsec)
    - P2 (King sigma): pixels (1 pixel = 2.357 arcsec)

    Parameters
    ----------
    axes : list of `MapAxis` or `MapAxes`
        Data axes, required are ["energy_true", "offset"].
    meta : dict
        Metadata dictionary.
    """
    tag = "psf_swift"
    required_axes = ["energy_true", "offset"]
    required_parameters = ["p0", "p1", "p2", "p3"]
    required_arguments = ["rad", "energy_true", "offset"]

    # Conversion factors
    PIXEL_TO_DEG = 2.357 * u.arcsec.to(u.deg)
    ARCMIN_TO_DEG = 1 / 60.0

    def __init__(self, axes, data, meta=None, unit=None):
        super().__init__(axes=axes, data=data, meta=meta, unit=unit)
        # Store coefficients for analytical evaluation
        # We expect 'coefficients' to be passed in meta or data, 
        # but ParametricPSF structure is rigid. 
        # We will store coefficients in a separate attribute if available.
        if meta and "coefficients" in meta:
             self.coefficients = meta["coefficients"]
        else:
             self.coefficients = None

    @classmethod
    def read(cls, filename, hdu="PC_PSF_COEF"):
        """Read PSF coefficients from FITS file."""
        print("calling SwiftPSF.read")
        with fits.open(filename) as hdul:
            table = hdul[hdu].data
            header = hdul[hdu].header
        
        # Extract coefficients
        # Structure: 4 rows (P0, P1, P2, P3), 5 columns (Name, C0, C1, C2, C3)
        # We'll store them in a dictionary or array
        coeffs = {}
        for row in table:
            name = row['PAR'].strip() # P0, P1, P2, P3
            coeffs[name] = {
                'c0': row['COEF0'],
                'c1': row['COEF1'],
                'c2': row['COEF2'],
                'c3': row['COEF3']
            }
        
        # Create dummy axes and data to satisfy ParametricPSF init
        # We don't use them for interpolation, but the base class needs them.
        energy_axis = MapAxis.from_energy_bounds("0.1 keV", "10 keV", nbin=1, name="energy_true")
        offset_axis = MapAxis.from_bounds(0, 1, nbin=1, unit="deg", name="offset")
        axes = MapAxes([energy_axis, offset_axis])
        
        # Create empty data array (structured array)
        dtype = {
            "names": cls.required_parameters,
            "formats": len(cls.required_parameters) * (np.float32,),
        }
        data = np.zeros(axes.shape, dtype=dtype)
        unit = {p: u.Unit("") for p in cls.required_parameters} # Units handled internally

        meta = dict(header)
        meta["coefficients"] = coeffs
        
        return cls(axes=axes, data=data, meta=meta, unit=unit)

    def evaluate_parameters(self, energy_true, offset):
        """Evaluate parameters P0-P3 analytically."""
        print("calling SwiftPSF.evaluate_parameters")
        if self.coefficients is None:
            raise ValueError("PSF coefficients not loaded.")

        # Convert inputs to units expected by the formula
        # Energy: PI (1 PI = 10 eV = 0.01 keV) -> 1 keV = 100 PI
        # where offaxis is in arcmin/0.1 and energy in keV/0.1 respectively. 
        # but doesn't matter as not term depends on them
        energy_pi = energy_true.to_value(u.keV) * 100
        energy_pi = energy_true.to_value(u.keV) / 0.1
        # Offset: arcmin
        offset_arcmin = offset.to_value(u.arcmin)
        offset_arcmin = offset.to_value(u.arcmin) / 0.1

        pars = {}
        for name in ["P0", "P1", "P2", "P3"]:
            c = self.coefficients[name]
            # P_i = C0 + C1*off + C2*en + C3*en*off
            # where offaxis is in arcmin/0.1 and energy in keV/0.1 respectively. 
            val = (c['c0'] + 
                   c['c1'] * offset_arcmin + 
                   c['c2'] * energy_pi + 
                   c['c3'] * energy_pi * offset_arcmin)
            
            # Store with appropriate units/conversions for evaluate_direct
            # P0: weight (dimensionless)
            # P1: Gauss sigma (pixels -> deg)
            # P3: King gamma (dimensionless)
            # P3: King sigma (pixels -> deg)
            
            key = name.lower()
            if name == "P0":
                pars[key] = val
                print("P0:", val[0][0][0][0])
            elif name == "P1":
                pars[key] = val * self.PIXEL_TO_DEG * u.deg
                print("P1 (pix):", val[0][0][0][0])
                print("P1:", val[0][0][0][0] * self.PIXEL_TO_DEG * u.deg)
            # elif name == "P2":
            #     pars[key] = val
            # elif name == "P3":
            #     pars[key] = val * self.PIXEL_TO_DEG * u.deg
            elif name == "P2":
                pars[key] = val * self.PIXEL_TO_DEG * u.deg
                print("P2 (pix):", val[0][0][0][0])
                print("P2:", val[0][0][0][0] * self.PIXEL_TO_DEG * u.deg)
            elif name == "P3":
                pars[key] = val
                print("P3:", val[0][0][0][0])
                
        return pars

    @staticmethod
    def evaluate_direct(rad, p0, p1, p2, p3):
        """Evaluate the PSF model.
        
        PSF = P0 * Gauss(P1) + (1 - P0) * King(P2, P3)
        """
        print("calling SwiftPSF.evaluate_direct")
        w = p0
        sigma = p1
        rc = p2
        beta = p3
        term_gauss = np.exp(-rad**2 / (2 * sigma**2))
        term_king = (1 + (rad / rc)**2) ** (-beta)
        return (w * term_gauss + (1 - w) * term_king)


    # @staticmethod
    # def evaluate_containment(rad, p0, p1, p2, p3):
    #     """Containment of the PSF at given axes coordinates.

    #     Parameters
    #     ----------
    #     rad : `~astropy.units.Quantity`
    #         Rad value.
    #     p0, p1, p2, p3 : `~astropy.units.Quantity`
    #         Swift PSF parameters (w, sigma, rc, beta).

    #     Returns
    #     -------
    #     containment : `~numpy.ndarray`
    #         Containment fraction (0 to 1).
    #     """
    #     w = p0
    #     sigma = p1
    #     rc = p2
    #     beta = p3

    #     with np.errstate(divide="ignore", invalid="ignore"):
    #         # Component 1: Gaussian
    #         gauss_cont = 2 * np.pi * w * (sigma**2) * (1 - np.exp(-rad**2 / (2 * sigma**2)))
            
    #         # Component 2: King
    #         king_cont = np.pi * (1 - w) * (rc**2 / (beta - 1)) * (1 - (1 + (rad / rc)**2) ** (1 - beta))
            
    #         containment = gauss_cont + king_cont

    #     return containment

    @staticmethod
    def evaluate_containment(rad, p0, p1, p2, p3):
        """Containment of the PSF at given axes coordinates.

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value.
        p0, p1, p2, p3 : `~astropy.units.Quantity`
            Swift PSF parameters (w, sigma, rc, beta).

        Returns
        -------
        containment : `~numpy.ndarray`
            Containment fraction (0 to 1).
        """
        w = p0
        sigma = p1
        rc = p2
        beta = p3

        with np.errstate(divide="ignore", invalid="ignore"):
            # Component 1: Gaussian
            gauss_cont = 2 * np.pi * w * (sigma**2) * (1 - np.exp(-rad**2 / (2 * sigma**2)))
            
            # Component 2: King
            king_cont = np.pi * (1 - w) * (rc**2 / (1-beta)) * ( ((1 + (rad / rc)**2) ** (1 - beta)) - 1)
            
            containment = gauss_cont + king_cont

        return containment
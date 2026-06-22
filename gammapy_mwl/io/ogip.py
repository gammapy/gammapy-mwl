# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from regions import Regions

from gammapy.utils.scripts import make_path, make_name
from gammapy.maps import RegionNDMap, MapAxis, RegionGeom, WcsGeom
from gammapy.irf import EDispKernel, EDispKernelMap
from gammapy.datasets import SpectrumDatasetOnOff,SpectrumDatasetChi2
#from .spectrumdataset import SpectrumDatasetChi2
from gammapy.data import GTI

__all__ = ["StandardOGIPDatasetReader"]


@classmethod
def from_hdulist(cls, hdulist, hdu1="MATRIX", hdu2="EBOUNDS"):
    """Create `EDispKernel` object from a FITS HDUList.

    Parameters
    ----------
    hdulist : `~astropy.io.fits.HDUList`
        HDU list containing the energy dispersion matrix and energy bounds.
    hdu1 : str, optional
        Name of the HDU with the energy dispersion matrix (default: "MATRIX").
    hdu2 : str, optional
        Name of the HDU with the energy axis information (default: "EBOUNDS").

    Returns
    -------
    edisp : `EDispKernel`
        Energy dispersion kernel instance.
    """
    matrix_hdu = hdulist[hdu1]
    ebounds_hdu = hdulist[hdu2]

    data = matrix_hdu.data
    header = matrix_hdu.header

    pdf_matrix = np.zeros([len(data), header["DETCHANS"]], dtype=np.float64)

    #check for TLMIN keyword to determine if indexing starts at 0 or 1:
    col_num = matrix_hdu.columns.names.index("F_CHAN") + 1
    ind_offset = matrix_hdu.header.get(f"TLMIN{col_num}", 0)

    for i, l in enumerate(data):
        if l.field("N_GRP"):
            m_start = 0
            for k in range(l.field("N_GRP")):
                if np.isscalar(l.field("N_CHAN")):
                    f_chan = l.field("F_CHAN")-ind_offset
                    n_chan = l.field("N_CHAN")
                else:
                    f_chan = l.field("F_CHAN")[k]-ind_offset
                    n_chan = l.field("N_CHAN")[k]
                try:
                    pdf_matrix[i, f_chan : f_chan + n_chan] = l.field("MATRIX")[m_start : m_start + n_chan]
                except IndexError:
                    # Handle single-channel case (e.g. UVOT uvot2pha rsp file)
                    pdf_matrix[i, f_chan : f_chan + n_chan] = np.asarray([l.field("MATRIX"),])[m_start : m_start + n_chan]
                m_start += n_chan

    table = Table.read(ebounds_hdu)
    energy_min = table["E_MIN"].quantity
    energy_max = table["E_MAX"].quantity
    # In some files, energies are reversed in the EBOUNDS HDU
    if energy_min[0] > energy_max[0]:
        energy_min, energy_max = energy_max, energy_min

    energy_edges = np.append(energy_min.value, energy_max.value[-1]) * energy_min.unit
    energy_axis = MapAxis.from_edges(energy_edges, name="energy", interp="lin")

    table = Table.read(matrix_hdu)
    energy_min = table["ENERG_LO"].quantity
    energy_max = table["ENERG_HI"].quantity
    # Avoid min edge being zero
    energy_min[0] += 1e-2 * (energy_max[0] - energy_min[0])
    energy_edges = np.append(energy_min.value, energy_max.value[-1]) * energy_min.unit
    energy_true_axis = MapAxis.from_edges(energy_edges, name="energy_true", interp="lin")

    return cls(axes=[energy_true_axis, energy_axis], data=pdf_matrix)


EDispKernel.from_hdulist = from_hdulist


class StandardOGIPDatasetReader:
    """Reader for OGIP-compliant spectral datasets.

    Reads PHA, ARF, RMF, and background files in OGIP format and constructs a
    `SpectrumDatasetOnOff` object.

    Parameters
    ----------
    filename : str or `~pathlib.Path`
        Path to the OGIP PHA file.
    region_hdu : str, optional
        Name of the HDU containing region information (default: "REGION").
    gti_hdu : str, optional
        Name of the HDU containing GTI information (default: "GTI").
    """

    tag = "ogip"

    def __init__(self, filename, region_hdu="REGION", gti_hdu="GTI"):
        self.filename = make_path(filename)
        self.region_hdu = region_hdu
        self.gti_hdu = gti_hdu

    def get_valid_path(self, filename):
        """Resolve a file path relative to the PHA file location.

        Parameters
        ----------
        filename : str or `Path`
            File name or path.

        Returns
        -------
        path : `Path`
            Absolute path to the file.
        """
        filename = make_path(filename)
        # Always resolve relative to the PHA file's directory
        if not filename.is_absolute():
            return self.filename.parent / filename
        # If absolute, check if it exists, else try relative
        if not filename.exists():
            return self.filename.parent / filename.name
        return filename

    def get_filenames(self, pha_meta):
        """Extract related file names from PHA metadata, resolving them relative to the PHA file.

        Parameters
        ----------
        pha_meta : dict
            Metadata from the PHA file.

        Returns
        -------
        filenames : dict
            Dictionary with keys "arffile", "rmffile" (optional), and "bkgfile" (optional).
        """
        filenames = {}
        if "ANCRFILE" in pha_meta:
            filenames["arffile"] = self.get_valid_path(pha_meta["ANCRFILE"])
        if "BACKFILE" in pha_meta:
            filenames["bkgfile"] = self.get_valid_path(pha_meta["BACKFILE"])
        if "RESPFILE" in pha_meta:
            filenames["rmffile"] = self.get_valid_path(pha_meta["RESPFILE"])
        return filenames

    def _read_regions(self, hdulist):
        """Read region data from an HDUList.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            HDU list to read from.

        Returns
        -------
        region : `~regions.CompoundSkyRegion` or None
            Compound region object, or None if not present.
        wcs : `~astropy.wcs.WCS` or None
            WCS object, or None if not present.
        """
        region, wcs = None, None
        if self.region_hdu in hdulist:
            region_table = Table.read(hdulist[self.region_hdu])
            pix_region = Regions.parse(region_table, format="fits")
            pix_region = pix_region.shapes.to_regions()
            wcs = WcsGeom.from_header(region_table.meta).wcs
            regions = [reg.to_sky(wcs) for reg in pix_region]
            region = list_to_compound_region(regions)
        return region, wcs

    def _read_gti(self, hdulist):
        """Read GTI table from an HDUList.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            HDU list to read from.

        Returns
        -------
        gti : `~gammapy.data.GTI` or None
            GTI object, or None if not present.
        """
        from astropy.time import Time
        gti = None
        if self.gti_hdu in hdulist:
            header = hdulist[self.gti_hdu].header
            gti_table = Table.read(hdulist[self.gti_hdu])
            ref_mjd = header['MJDREFI'] + header['MJDREFF']
            if gti_table['START'].unit == 's':
                gti_table['START'] = Time(ref_mjd + gti_table['START'].to('s').value / 86400., format='mjd')
                gti_table['STOP'] = Time(ref_mjd + gti_table['STOP'].to('s').value / 86400., format='mjd')
            gti = GTI(gti_table)
        return gti

    @staticmethod
    def extract_spectrum(pha_table):
        """Extract spectrum data from a PHA table.

        The input table must follow OGIP format. Only counts spectra are supported.

        Parameters
        ----------
        pha_table : `~astropy.table.Table`
            Table containing the PHA data.

        Returns
        -------
        spectrum_data : dict
            Dictionary with spectrum data and metadata.
        """
        spectrum_data = {}
        pha_meta = pha_table.meta
        is_rate_pha = False
        
        if pha_meta.get("HDUCLASS") != "OGIP":
            raise ValueError("Input file is not an OGIP file.")
        if pha_meta.get("HDUCLAS1") != "SPECTRUM":
            raise ValueError("Input file is not a PHA file.")
        if pha_meta.get("HDUCLAS2") == "NET":
            raise ValueError("Subtracted PHA files are not supported.")
        if pha_meta.get("HDUCLAS3") == "RATE":
            is_rate_pha = True
            #raise ValueError("Rate PHA files are not supported.")
        if pha_meta.get("HDUCLAS4") == "TYPE:II":
            raise ValueError("Type II PHA files are not supported.")

        spectrum_data["livetime"] = pha_meta["EXPOSURE"] * u.s
        spectrum_data["channel"] = pha_table["CHANNEL"]
        
        if is_rate_pha:
            pha_table['COUNTS'] = (pha_table['RATE'] * pha_meta["EXPOSURE"] * u.s).to(u.ct)
            if "STAT_ERR" in pha_table.colnames:
                pha_table['STAT_ERR'] = (pha_table['STAT_ERR'] * pha_meta["EXPOSURE"] * u.s).to(u.ct)
        
        spectrum_data["counts"] = pha_table["COUNTS"]
        if "STAT_ERR" in pha_table.colnames:
            spectrum_data["sigma"] = pha_table["STAT_ERR"]

        mask_safe = True
        if "QUALITY" in pha_table.columns:
            mask_safe = pha_table["QUALITY"].data == 0
        spectrum_data["mask_safe"] = mask_safe

        #grouping = pha_table["GROUPING"] if "GROUPING" in pha_table.columns else None
        #spectrum_data["grouping"] = grouping

        if "BACKSCAL" in pha_table.columns:
            acceptance = pha_table["BACKSCAL"]
        else:
            acceptance = pha_meta["BACKSCAL"]
        spectrum_data["acceptance"] = acceptance

        exposure = pha_meta["EXPOSURE"]
        spectrum_data["acceptance"] *= exposure

        area_scale = pha_table["AREASCAL"] if "AREASCAL" in pha_table.columns else 1
        spectrum_data["area_scale"] = area_scale

        return spectrum_data

    def read(self, filenames=None, name=None):
        """Read OGIP files and return a `SpectrumDatasetOnOff`.

        Parameters
        ----------
        filenames : dict, optional
            Dictionary with file paths for "arffile", "rmffile", and "bkgfile".
            If not provided, will be inferred from the PHA file metadata.
        name : str, optional
            Name for the dataset.

        Returns
        -------
        dataset : `~gammapy.datasets.SpectrumDatasetOnOff`
            The constructed spectral dataset.
        """
        hdulist = fits.open(self.filename, memmap=False)
        pha_table = Table.read(hdulist["spectrum"])

        data = self.extract_spectrum(pha_table)
        region, wcs = self._read_regions(hdulist)
        gti = self._read_gti(hdulist)
        
        if filenames is None:
            filenames = self.get_filenames(pha_meta=pha_table.meta)

        edisp_kernel = EDispKernel.read(filenames["rmffile"])
        energy_axis = edisp_kernel.axes["energy"]
        energy_true_axis = edisp_kernel.axes["energy_true"]

        # Some instruments, e.g. UVOT and BAT, do not have separate rmffile and arffile, but a respfile.
        if str(filenames["arffile"]).split("/")[-1] != "NONE":
            arf_table = Table.read(filenames["arffile"], hdu="SPECRESP")

        if filenames["bkgfile"]!=None:           
            bkg_table = Table.read(filenames["bkgfile"])
            data_bkg = self.extract_spectrum(bkg_table)
        else:
            excess = np.asarray(data["counts"].data)
            sigma  = np.asarray(data["sigma"].data)
            
        geom = RegionGeom(region=region, wcs=wcs, axes=[energy_axis])

        counts = RegionNDMap(geom=geom, data=data["counts"].data, unit="")
        acceptance = RegionNDMap(geom=geom, data=data["acceptance"], unit="")
        mask_safe = RegionNDMap(geom=geom, data=data["mask_safe"], unit="")

        if filenames["bkgfile"]!=None:
            counts_off = RegionNDMap(geom=geom, data=data_bkg["counts"].data, unit="")
            acceptance_off = RegionNDMap(geom=geom, data=data_bkg["acceptance"], unit="")
        
        geom_true = RegionGeom(region=region, wcs=wcs, axes=[energy_true_axis])

        if str(filenames["arffile"]).split("/")[-1] != "NONE":
            # Standard case: ARF table contains the response
            exposure = arf_table["SPECRESP"].quantity * data["livetime"]
        else:
            # Extract exposure and migration matrix from the RSP file
            exposure = edisp_kernel.data * u.Unit("cm2") * data["livetime"]
            edisp_kernel.data = 1

        # Squeeze singleton dimensions, but keep the correct number of bins
        exposure_array = np.squeeze(np.asarray(exposure))
        n_bins = len(energy_true_axis.center)
        if exposure_array.shape[0] != n_bins:
            raise ValueError(
                f"Exposure shape {exposure_array.shape} does not match number of energy bins {n_bins}."
            )
        exposure = RegionNDMap(geom=geom_true, data=exposure_array, unit=getattr(exposure, "unit", ""))

        edisp = EDispKernelMap.from_edisp_kernel(edisp_kernel, geom=exposure.geom)

        #index = np.where(data["grouping"] == 1)[0]
        #if len(index) != 0:
        #    edges = np.append(energy_axis.edges[index], energy_axis.edges[-1])
        #    grouping_axis = MapAxis.from_energy_edges(edges, interp=energy_axis._interp)
        #else:
        #    grouping_axis = energy_axis

        name = make_name(name)
        if filenames["bkgfile"]==None:
            sigma = RegionNDMap(geom=geom, data=data["sigma"].data, unit="")
            dataset = SpectrumDatasetChi2(
                name=name,
                counts=counts,
                sigma=sigma,
                edisp=edisp,
                exposure=exposure,
                mask_safe=mask_safe,
                gti=gti,
                meta_table=pha_table.meta,
            )
        else:
            dataset = SpectrumDatasetOnOff(
                name=name,
                counts=counts,
                acceptance=acceptance,
                counts_off=counts_off,
                acceptance_off=acceptance_off,
                edisp=edisp,
                exposure=exposure,
                mask_safe=mask_safe,
                gti=gti,
                meta_table=pha_table.meta,
            )

        if "livetime" not in dataset.exposure.meta:
           dataset.exposure.meta['livetime'] = data["livetime"]

        return dataset

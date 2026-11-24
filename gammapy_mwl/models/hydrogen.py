# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Module to calculate Galactic HI column density (nh) and generate/load TBabs absorption models.

To manually query/obtain the Galactic HI column density (nh) value for a specific coordinate:
1. Use the Swift nh tool: https://www.swift.ac.uk/analysis/nhtot/
2. Use HEASARC's nh tool: https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3nh/w3nh.pl
3. Use the nh tool within the astropy/astroquery package (e.g. astroquery.heasarc).
"""

import logging
from pathlib import Path

import astropy.units as u
from astropy.table import Table
import numpy as np

from gammapy.maps import MapAxis, RegionNDMap
from gammapy.modeling.models import TemplateNDSpectralModel

logger = logging.getLogger(__name__)

# Determine paths relative to this file's location
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_TBABS_FILE = MODULE_DIR / "data" / "tbabs_tau_factor_vs_nh_energy.ecsv"


def generate_tbabs_interp_table(outfile):
    """Generates the 2D interpolation ECSV table using local Sherpa/XSpec installation.
    The default model is Tbabs (Wilms et al 2000, doi: 10.1086/317016). 
    """
    from gammapy_mwl.models.sherpa import SherpaSpectralModel
    from sherpa.astro.xspec import XSTBabs

    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nh_array = np.round(np.linspace(-4, 2.5, 10 * 60), decimals=5)
    en_array = np.round(np.linspace(-2, 2.5, 10 * 10), decimals=5)

    tbabs_table_data = []

    for nh in nh_array:
        model = XSTBabs()
        model.nh.val = 10**nh
        wrapped_model = SherpaSpectralModel(model, default_units=(u.keV, 1))

        # Calculate optical depth tau
        tau = -np.log(wrapped_model(10**en_array * u.keV))
        tau = np.where(np.isinf(tau), 100.0, tau)
        tau = np.where(tau < 0, 0.0, tau)

        tbabs_table_data.append(list(np.round(tau, 5)))

    tbabs_table = Table(data=tbabs_table_data)
    tbabs_table.meta["log10_E_type"] = "rows"
    tbabs_table.meta["log10_E_values"] = list(en_array)
    tbabs_table.meta["log10_nh_type"] = "columns"
    tbabs_table.meta["log10_nh_values"] = list(nh_array)
    tbabs_table.meta["table data"] = "tau factor (natural log of absorption)"

    tbabs_table.write(out_path, format="ascii.ecsv", overwrite=True)
    logger.info(f"Interpolation table written to {out_path}")



def get_tbabs_template_model(
    nh=None,
    tbabsfile=None,
    freeze=True,
    outfile=None,
):
    """Creates a Gammapy TemplateNDSpectralModel from a pregenerated table of absorption.
    The default model is Tbabs (Wilms et al 2000, doi: 10.1086/317016).
    Uses relative directory pathways to find the data files if not explicitly provided.

    Parameters
    ----------
    nh : float or `~astropy.units.Quantity`, optional
        Galactic HI column density. If float, cm-2 unit is assumed.
    tbabsfile : str or `~pathlib.Path`, optional
        Path to the interpolation ECSV file.
    freeze : bool, optional
        Whether to freeze the nh parameter. Default is True.
    outfile : str or `~pathlib.Path`, optional
        Path to yaml file to write model to. Required to serialise the model.
    """
    # Use relative path if none provided
    if tbabsfile is None:
        tbabsfile = DEFAULT_TBABS_FILE
    else:
        tbabsfile = Path(tbabsfile)

    if not tbabsfile.exists():
        raise FileNotFoundError(
            f"Absorption template file not found at: {tbabsfile.resolve()}"
        )

    from astropy.table import Table
    tbabs_table = Table.read(tbabsfile, format="ascii.ecsv")

    # Clean multi-dimensional array extraction from ECSV columns
    tbabs_data = np.stack(
        [tbabs_table[col].astype(np.float64) for col in tbabs_table.colnames],
        axis=1
    )

    log_nh_array_20 = np.asarray(tbabs_table.meta["log10_nh_values"])
    log_en_array = np.asarray(tbabs_table.meta["log10_E_values"])

    energy_axis = MapAxis.from_nodes(
        10**log_en_array * u.keV, name="energy_true", interp="log"
    )
    nh_axis = MapAxis.from_nodes(
        10**log_nh_array_20 * 1e22 * u.Unit("cm-2"), name="nh", interp="log"
    )

    # Convert optical depth to transmission, preserving numerical thresholds
    transmission = np.exp(-np.transpose(tbabs_data))
    transmission = np.clip(transmission, 1e-5, None)

    region_ndmap = RegionNDMap.create(
        region=None,
        axes=[energy_axis, nh_axis],
        data=transmission,
    )

    if outfile is None:
        template_abs_model = TemplateNDSpectralModel(
            map=region_ndmap,
            interp_kwargs={"method": "linear", "fill_value": 1e-5},
        )

        if nh is not None:
            template_abs_model.nh.quantity = u.Quantity(nh, "cm-2")
            if freeze:
                template_abs_model.nh.frozen = True
        else:
            logger.warning(
                "Generating generic template absorption model with default nh parameters."
            )

    else:
        template_abs_model = TemplateNDSpectralModel(
            map=region_ndmap,
            filename=outfile,
            interp_kwargs={"method": "linear", "fill_value": 1e-5},
        )
        if nh is not None:
            template_abs_model.nh.quantity = u.Quantity(nh, "cm-2")
            if freeze:
                template_abs_model.nh.frozen = True
        else:
            logger.warning(
                "Generating generic template absorption model with default nh parameters."
            )
        template_abs_model.write(filename=outfile, overwrite=True)
        logger.info(f"Absorption model written to {outfile}")

    return template_abs_model

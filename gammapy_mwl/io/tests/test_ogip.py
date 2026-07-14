import pytest
import warnings
from gammapy_mwl.io.ogip import StandardOGIPDatasetReader
from gammapy.datasets import SpectrumDatasetOnOff
from gammapy.modeling.models import Models, SkyModel, PowerLawSpectralModel
from astropy import units as u
from astropy.io import fits
import numpy as np
from numpy.testing import assert_allclose

# Example template for expected shapes and sums (fill in values as needed)
EXPECTED_SHAPES = {
    "data/swift/xrt/swift_xrt_00036384074_src.pha": {
        "counts": (1024,1,1),
        "acceptance": (1024,1,1),
        "counts_off": (1024,1,1),
        "acceptance_off": (1024,1,1),
        "edisp": (2400, 1024, 1, 1),
        "exposure": (2400, 1, 1),
    },
    "data/swift/uvot/sw00036384074uvv_sk.src.pha": {
        "counts": (1,1,1),
        "acceptance": (1,1,1),
        "counts_off": (1,1,1),
        "acceptance_off": (1,1,1),
        "edisp": (1282, 1, 1, 1),
        "exposure": (1282, 1, 1),
    },
    "data/xmm/mos1/xmm_mos1_0605960101_src.pha": {
        "counts": (800,1,1),
        "acceptance": (800,1,1),
        "counts_off": (800,1,1),
        "acceptance_off": (800,1,1),
        "edisp": (2389, 800, 1, 1),
        "exposure": (2389, 1, 1),
    },
    "data/xmm/mos2/xmm_mos2_0605960101_src.pha": {
        "counts": (800,1,1),
        "acceptance": (800,1,1),
        "counts_off": (800,1,1),
        "acceptance_off": (800,1,1),
        "edisp": (2389, 800, 1, 1),
        "exposure": (2389, 1, 1),
    },
    "data/xmm/pn/xmm_pn_0605960101_src.pha": {
        "counts": (4096,1,1),
        "acceptance": (4096,1,1),
        "counts_off": (4096,1,1),
        "acceptance_off": (4096,1,1),
        "edisp": (2056, 4096, 1, 1),
        "exposure": (2056, 1, 1),
    },
    "data/erosita/em01_072135_820_SourceSpec_00001_c010.fits": {
        "counts": (1024,1,1),
        "acceptance": (1024,1,1),
        "counts_off": (1024,1,1),
        "acceptance_off": (1024,1,1),
        "edisp": (1024, 1024, 1, 1),
        "exposure": (1024, 1, 1),
    },
    "data/chandra/acis/circle30arcs.pi": {
        "counts": (1024,1,1),
        "acceptance": (1024,1,1),
        "counts_off": (1024,1,1),
        "acceptance_off": (1024,1,1),
        "edisp": (1070, 1024, 1, 1),
        "exposure": (1070, 1, 1),
    },
}

EXPECTED_SUMS = {
    "data/swift/xrt/swift_xrt_00036384074_src.pha": {
        "counts": 384,
        "counts_off": 280,
        "acceptance": 3790,
        "acceptance_off": 262303,
        "edisp": 1090,
        "exposure": 445171700.0,
    },
    "data/swift/uvot/sw00036384074uvv_sk.src.pha": {
        "counts": 3135,
        "counts_off": 16437,
        "acceptance": 143370,
        "acceptance_off": 1491009,
        "edisp": 1282,
        "exposure": 207368,
    },
    "data/xmm/mos1/xmm_mos1_0605960101_src.pha": {
        "counts": 33497,
        "counts_off": 2.369e5,
        "acceptance": 8.7e14,
        "acceptance_off": 8.7e16,
        "edisp": 2389,
        "exposure": 17992516000.0,
    },
    "data/xmm/mos2/xmm_mos2_0605960101_src.pha": {
        "counts": 31941,
        "counts_off": 225753,
        "acceptance": 8.744e14,
        "acceptance_off": 8.744e+16,
        "edisp": 2389,
        "exposure": 17992516000,
    },
    "data/xmm/pn/xmm_pn_0605960101_src.pha": {
        "counts": 70680,
        "counts_off": 333378,
        "acceptance": 2.5479e15,
        "acceptance_off": 2.5479e+17,
        "edisp": 2056,
        "exposure": 22077673000,
    },
    "data/erosita/em01_072135_820_SourceSpec_00001_c010.fits": {
        "counts": 6013,
        "counts_off": 3940,
        "acceptance": 3089.0511876683663,
        "acceptance_off": 248401.46840409856,
        "edisp": 1024,
        "exposure": 91403235.31326437,
    },
    "data/chandra/acis/circle30arcs.pi": {
        "counts": 4348,
        "acceptance": 6770.849424893879,
        "counts_off": 33190,
        "acceptance_off": 88368.86537272263,
        "edisp": 1070.0247,
        "exposure": 236322.58,
    },
}

@pytest.mark.parametrize(
    "pha_path",
    [
        "data/swift/xrt/swift_xrt_00036384074_src.pha",
        "data/swift/uvot/sw00036384074uvv_sk.src.pha",
        "data/xmm/mos1/xmm_mos1_0605960101_src.pha",
        "data/xmm/mos2/xmm_mos2_0605960101_src.pha",
        "data/xmm/pn/xmm_pn_0605960101_src.pha",
        "data/erosita/em01_072135_820_SourceSpec_00001_c010.fits",
        # "data/xmm/om/xmm_om_0605960101_src.pha",
        # "data/fermi/lat/fermi_lat_00036384074_src.pha",
        # "data/fermi/gbm/fermi_gbm_00036384074_src.pha",
    ]
)
def test_standard_ogip_dataset_reader_runs(pha_path):
    """Test that StandardOGIPDatasetReader returns a SpectrumDatasetOnOff for a Swift/XRT or UVOT OGIP PHA file,
    and that OGIP keywords and data arrays are read correctly and are not empty or zero."""
    # Suppress Astropy warnings for test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(pha_path) as hdulist:
            header = hdulist[1].header  # Usually the spectrum is in extension 1
            ogip_class = header.get("HDUCLASS")
            ogip_clas1 = header.get("HDUCLAS1")
            exposure = header.get("EXPOSURE")
            assert ogip_class == "OGIP"
            assert ogip_clas1 == "SPECTRUM"
            assert exposure is not None

        reader = StandardOGIPDatasetReader(pha_path)
        dataset = reader.read()
        assert isinstance(dataset, SpectrumDatasetOnOff)

        # Check attributes exist
        for attr in ["counts", "acceptance", "counts_off", "acceptance_off", "edisp", "exposure"]:
            value = getattr(dataset, attr)
            assert value is not None

        # Check shapes of arrays
        shape_map = {
            "counts": dataset.counts.data,
            "counts_off": dataset.counts_off.data,
            "acceptance": dataset.acceptance.data,
            "acceptance_off": dataset.acceptance_off.data,
            "edisp": dataset.edisp.edisp_map.data,
            "exposure": dataset.exposure.data,
        }
        for key, arr in shape_map.items():
            expected_shape = EXPECTED_SHAPES[pha_path][key]
            assert arr.shape == expected_shape, f"{key} shape {arr.shape} != {expected_shape}"
        
        sum_map = {
            "counts": dataset.counts.data,
            "counts_off": dataset.counts_off.data,
            "acceptance": dataset.acceptance.data,
            "acceptance_off": dataset.acceptance_off.data,
            "edisp": dataset.edisp.edisp_map.data,
            "exposure": dataset.exposure.data,
        }
        for key, arr in sum_map.items():
            expected_sum = EXPECTED_SUMS[pha_path][key]
            assert_allclose(arr.sum(), expected_sum, rtol=1e-2, 
                            err_msg=f"{key} sum {arr.sum()} != {expected_sum}")
        
        # Check arrays are not empty and do not sum to zero
        assert dataset.counts.data.sum() > 0
        assert dataset.counts_off.data.sum() > 0
        assert dataset.acceptance.data.sum() > 0
        assert dataset.acceptance_off.data.sum() > 0

        # Check that the exposure and livetime are read correctly from the dataset metadata
        meta = dataset.meta_table
        assert_allclose(meta["EXPOSURE"], exposure, rtol=1e-3)

        # Check that the axes align and an npred can be calculated
        m = SkyModel(spectral_model=PowerLawSpectralModel(amplitude=1*u.Unit("keV-1 cm-2 s-1"),reference=1*u.keV))
        dataset.models=Models([m])
        assert np.sum(dataset.npred_signal().data) > 0
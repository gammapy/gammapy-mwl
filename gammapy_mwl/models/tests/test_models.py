import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u

from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy_mwl.models.sherpa import SherpaSpectralModel


def test_SherpaSpectralModel():
    sherpa = pytest.importorskip("sherpa")
    from sherpa.models import basic

    energy_grid = np.linspace(0.5, 10.0, 10) * u.keV
    plaw = basic.PowLaw1D()
    plaw.ampl = 1e-3
    plaw.gamma = 2

    #abs_model = sherpa.astro.xspec.XSwabs()
    #abs_model.nH = 5

    # Gammapy wrapper
    f1 = SherpaSpectralModel(plaw, integrated=False)
    #f2 = SherpaSpectralModel(abs_model, default_units=(u.keV, 1))
    f3 = f1  #* f2

    # Plain sherpa
    plaw_with_abs = plaw #* abs_model

    assert_allclose(f3(energy_grid).value[:-1], plaw_with_abs(energy_grid.value)[:-1])
    assert_allclose(f3.evaluate(energy_grid,2,1,1e-3,5).value[:-1], plaw_with_abs(energy_grid.value)[:-1])

    SkyModel(spectral_model=f3)  # Test evaluate on simple geom
    #with pytest.raises(AttributeError):
    #    SkyModel(spectral_model=f2)  # Wrong units, f2 is an absorption model


def test_tbabs_model():
    from gammapy_mwl.models.hydrogen import get_tbabs_template_model
    
    nh = 2e21 * u.Unit("cm-2")
    model = get_tbabs_template_model(nh=nh)
    
    assert model.tag[0] == "TemplateNDSpectralModel"
    assert_allclose(model.nH.quantity, nh)
    assert model.nH.frozen is True
    
    # Test evaluation
    energy = [1, 2, 10] * u.keV
    transmission = model(energy)
    assert len(transmission) == 3
    assert np.all(transmission >= 0) and np.all(transmission <= 1)


def test_xredden_model():
    from gammapy_mwl.models.dustextinction import get_xredden_template_model
    
    ebv = 0.2
    model = get_xredden_template_model(ebv=ebv)
    
    assert model.tag[0] == "TemplateNDSpectralModel"
    assert_allclose(model.ebv.value, ebv)
    assert model.ebv.frozen is True
    
    # Test evaluation
    energy = [1, 2, 10] * u.keV
    transmission = model(energy)
    assert len(transmission) == 3
    assert np.all(transmission >= 0) and np.all(transmission <= 1)


@pytest.mark.skip(reason="Xspec models not in CI yet")
def test_sherpa_tbabs_vs_table():
    """Check that the Sherpa XSTBabs model broadly agrees with the interpolation table."""
    pytest.importorskip("sherpa")
    from gammapy_mwl.models.hydrogen import sherpa_xtbabs_model, get_tbabs_template_model

    nh = 2e21 * u.Unit("cm-2")
    sherpa_model = sherpa_xtbabs_model(nh=nh)
    table_model = get_tbabs_template_model(nh=nh)

    # Test on a grid away from the boundaries to avoid interpolation edge effects
    energy = np.logspace(np.log10(0.3), np.log10(10), 20) * u.keV
    t_sherpa = sherpa_model(energy).value
    t_table = table_model(energy).value

    # Values should agree to within 5% (interpolation residuals are expected)
    assert_allclose(t_sherpa, t_table, rtol=0.05)

@pytest.mark.skip(reason="Xspec models not in CI yet")
def test_sherpa_xredden_vs_table():
    """Check that the Sherpa XSxredden model broadly agrees with the interpolation table."""
    pytest.importorskip("sherpa")
    from gammapy_mwl.models.dustextinction import sherpa_xredden_model, get_xredden_template_model

    ebv = 0.2
    sherpa_model = sherpa_xredden_model(ebv=ebv)
    table_model = get_xredden_template_model(ebv=ebv)

    # Test on a grid away from the boundaries to avoid interpolation edge effects
    energy = np.logspace(np.log10(0.3), np.log10(10), 20) * u.keV
    t_sherpa = sherpa_model(energy).value
    t_table = table_model(energy).value

    # Values should agree to within 5% (interpolation residuals are expected)
    assert_allclose(t_sherpa, t_table, rtol=0.05)


def test_SherpaSpectralModel_multicomponent():
    # test multicomponent wrapping of sherpa models
    sherpa = pytest.importorskip("sherpa")
    from sherpa.models import basic

    energy_grid = np.linspace(0.5, 10.0, 10) * u.keV
    plaw = basic.PowLaw1D()
    plaw.ampl = 1e-3
    plaw.gamma = 2

    #abs_model = sherpa.astro.xspec.XSwabs()
    #abs_model.nH = 5
    plaw2 = basic.PowLaw1D()
    plaw2.ampl = 5e-3
    plaw2.gamma = 3

    # multicomponent sherpa model:

    sherpa_model = plaw + plaw2

    # Gammapy wrapper
    gammapy_model = SherpaSpectralModel(sherpa_model, integrated=False)

    assert_allclose(gammapy_model(energy_grid).value[:-1], sherpa_model(energy_grid.value)[:-1])
    assert_allclose(gammapy_model.evaluate(energy_grid,2,1,1e-3,5).value[:-1], sherpa_model(energy_grid.value)[:-1])

    m = SkyModel(spectral_model=gammapy_model)  # Test evaluate on simple geom

    # Check that parameter names are unique
    for i in m.parameters.names:
        if m.parameters.names.count(i) > 1:
            pytest.raises(AttributeError)

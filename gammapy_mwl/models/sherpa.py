import astropy.units as u
import numpy as np
from gammapy.modeling.models import SpectralModel
from gammapy.modeling import Parameter, Parameters


class SherpaSpectralModel(SpectralModel):
    """A wrapper for Sherpa spectral models.

    Parameters
    ----------
    sherpa_model :
        An instance of the models defined in `~sherpa.models` or `~sherpa.astro.xspec`.
    integrated:
        Set to True for correct evaluation of `sherpa.astro.xspec.XSAdditiveModel` models (additive XSpec models)
        or more-component models containing an additive XSpec model (e.g. apec, TBabs*apec).
        False for other models, such as `sherpa.models.basic` (e.g. PowLaw1D)
        or `sherpa.astro.xspec.XSMultiplicativeModel` models (e.g. TBabs). Default is True.
    default_units : tuple
        Units of the input energy array and output model evaluation (find them in the sherpa/xspec docs!)
    """

    tag = ["SherpaSpectralModel", "sherpa", "xspec"]

    def __init__(
        self, sherpa_model, integrated=True, default_units=(u.keV, 1 / (u.keV * u.cm ** 2 * u.s))
    ):
        self.sherpa_model = sherpa_model
        self.integrated = integrated
        self.default_units = default_units
        self.default_parameters = self._wrap_parameters()
        super().__init__()

    def _wrap_parameters(self):
        parameters = []
        self._remove_duplicate_parameter_names()
        for par in self.sherpa_model.pars:
            parameter = Parameter(
                name=par.name, value=par.val, frozen=par.frozen, min=par.min, max=par.max
            )
            # TODO: set unit?
            parameters.append(parameter)
        return Parameters(parameters)

    def _remove_duplicate_parameter_names(self):
        names = [par.name for par in self.sherpa_model.pars]
        for i, par in enumerate(self.sherpa_model.pars):
            if names.count(par.name)>1:
                if names[:i+1].count(par.name)>1:
                    par.name = par.name+str(names[:i].count(par.name))

    def _update_sherpa_parameters(self, **kwargs):
        """Update sherpa model parameters"""
        for name, value in kwargs.items():
            self.sherpa_model.pars[[x.name for x in self.sherpa_model.pars].index(name)].val=value

    def evaluate(self, energy, *args):
        if not isinstance(energy, u.Quantity):
            raise ValueError("The energy must be a Quantity object.")
        else:
            energy = energy.to(self.default_units[0])

        # Trickeries due to the sherpa model evaluation scheme
        # (https://sherpa.readthedocs.io/en/4.14.1/evaluation/index.html)
        energy = np.array(energy)
        shape = energy.shape
        energy = energy.flatten()
        energy = np.append(energy, energy[-1] * 2)

        kwargs = {name: q for name, q in zip(self.default_parameters.names, args)}
        self._update_sherpa_parameters(**kwargs)

        y_ = self.sherpa_model(energy)[:-1]
        if self.integrated:
            y_ /= energy[1:] - energy[:-1]
        y_ = y_ * self.default_units[1]

        return y_.reshape(shape)

    def __call__(self, energy):
        kwargs = {par.name: par.quantity for par in self.parameters}
        kwargs = self._convert_evaluate_unit(kwargs, energy)
        args = list(kwargs.values())
        return self.evaluate(energy, *args)

"""Categorical lineage likelihood with alternative ancestral localities."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def prior_precision(scales, categories):
    """Precision for category-symmetric Gaussian coefficient differences.

    The first category is the reference. Non-reference coefficients have
    variance ``scale**2`` and pairwise covariance ``scale**2 / 2``. Vectors
    store all design coefficients for category 1, then category 2, and so on.
    """
    scales = np.asarray(scales, dtype=float)
    if categories < 2 or scales.ndim != 1 or not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError('Positive coefficient scales and at least two categories are required')
    contrast = 2 * (np.eye(categories - 1) - np.ones((categories - 1, categories - 1)) / categories)
    return np.kron(contrast, np.diag(1 / scales**2))


class LabelMixture:
    """One categorical observation per unit, averaged over its localities.

    Rows of ``x`` must group each unit's alternative localities consecutively.
    ``counts`` gives the number of equally weighted localities for each unit;
    ``y`` gives its zero-based category. Optional nonnegative ``unit_weights``
    weight whole units for a declared sensitivity analysis. They do not turn
    the units into verified independent genealogies or adjust for selection.
    """

    def __init__(self, x, y, counts, scales, categories, unit_weights=None):
        self.x = np.asarray(x, dtype=float)
        labels, sizes = np.asarray(y), np.asarray(counts)
        if self.x.ndim != 2 or not np.isfinite(self.x).all():
            raise ValueError('The design must be a finite matrix')
        if labels.ndim != 1 or sizes.shape != labels.shape or len(labels) == 0:
            raise ValueError('One category and locality count are required per unit')
        if not np.all(np.isfinite(labels) & (labels == np.floor(labels))) or not np.all((labels >= 0) & (labels < categories)):
            raise ValueError('Categories must be integers within the declared range')
        if not np.all(np.isfinite(sizes) & (sizes == np.floor(sizes)) & (sizes > 0)):
            raise ValueError('Locality counts must be positive integers')
        self.y, self.counts = labels.astype(int), sizes.astype(int)
        self.n, self.p = self.x.shape
        self.k = categories - 1
        if self.counts.sum() != self.n or len(scales) != self.p:
            raise ValueError('Locality counts and prior scales must match the design')
        self.q = prior_precision(scales, categories)
        self.starts = np.r_[0, np.cumsum(self.counts)[:-1]]
        self.index = np.repeat(np.arange(len(self.y)), self.counts)
        self.local_y = self.y[self.index]
        self.indicators = (self.local_y[:, None] == np.arange(1, categories)).astype(float)
        self.unit_weights = np.ones(len(self.y)) if unit_weights is None else np.asarray(unit_weights, dtype=float)
        if self.unit_weights.shape != self.y.shape or not np.all(np.isfinite(self.unit_weights) & (self.unit_weights >= 0)) or not self.unit_weights.sum() > 0:
            raise ValueError('Unit weights must be finite, nonnegative and not all zero')

    def evaluate(self, coefficients, hessian=False):
        """Return negative log density, gradient and optionally its Hessian.

        Density omits the Gaussian normalization constant, which does not
        depend on the coefficients. The Hessian is observed information;
        its inverse supplies a local approximation, not sampled uncertainty.
        """
        v = np.asarray(coefficients, dtype=float)
        if v.shape != (self.k * self.p,) or not np.isfinite(v).all():
            raise ValueError('Coefficient vector has invalid size or values')
        eta = self.x @ v.reshape(self.k, self.p).T
        logits = np.column_stack([np.zeros(self.n), eta])
        lp = logits - logsumexp(logits, axis=1)[:, None]
        observed = lp[np.arange(self.n), self.local_y]
        maximum = np.maximum.reduceat(observed, self.starts)
        exponential = np.exp(observed - maximum[self.index])
        total = np.add.reduceat(exponential, self.starts)
        responsibility = exponential / total[self.index]
        local_weight = responsibility * self.unit_weights[self.index]
        probability = np.exp(lp[:, 1:])
        qv = self.q @ v
        value = -self.unit_weights @ (maximum + np.log(total) - np.log(self.counts)) + .5 * v @ qv
        gradient = -((local_weight[:, None] * (self.indicators - probability)).T @ self.x).ravel() + qv
        if not hessian:
            return float(value), gradient

        information = self.q.copy()
        for a in range(self.k):
            for b in range(self.k):
                weight = local_weight * probability[:, a] * ((a == b) - probability[:, b])
                information[a*self.p:(a+1)*self.p, b*self.p:(b+1)*self.p] += self.x.T @ (weight[:, None] * self.x)
        scores = ((self.indicators - probability)[:, :, None] * self.x[:, None, :]).reshape(self.n, -1)
        group_scores = np.add.reduceat(responsibility[:, None] * scores, self.starts, axis=0)
        information -= scores.T @ (local_weight[:, None] * scores)
        information += group_scores.T @ (self.unit_weights[:, None] * group_scores)
        return float(value), gradient, (information + information.T) / 2

    def optimize(self, start=None, max_iterations=1000, gradient_tolerance=1e-5):
        """Find a checked conditional mode and local covariance approximation."""
        start = np.zeros(self.k * self.p) if start is None else np.asarray(start, dtype=float)
        result = minimize(self.evaluate, start, jac=True, method='BFGS',
                          options={'gtol': 1e-8, 'maxiter': max_iterations})
        value, gradient, information = self.evaluate(result.x, hessian=True)
        curvature = float(np.linalg.eigvalsh(information).min())
        error = float(np.abs(gradient).max())
        if error >= gradient_tolerance or curvature <= 0 or not np.isfinite(result.x).all():
            raise ValueError(f'Unresolved mode: gradient={error}, minimum curvature={curvature}')
        return dict(mode=result.x, covariance=np.linalg.inv(information), objective=value,
                    gradient=error, curvature=curvature, iterations=int(result.nit))


def make_designs(cohort, measurements, configuration, draw=None, quadratic=False, quadratic_reference=None):
    """Build joint period designs using frozen controls and regional centres.

    All components and periods enter the same likelihood. Component units stay
    on their common reference scale. Regionally centred scores use the supplied
    centres, including in measurement and whole-unit omission sensitivities.
    The caller freezes the configuration before inspecting new coefficients.
    """
    from scipy.linalg import helmert

    local = np.asarray(cohort['local'])
    counts = np.asarray(cohort['counts'])
    weights = np.repeat(1 / counts, counts)
    periods, components = configuration['periods'], configuration['components']
    dimension = len(periods) * len(components)
    scores = measurements['scores'] if draw is None else measurements['score_draws'][draw]
    recording = measurements['recording_draws'].mean(0) if draw is None else measurements['recording_draws'][draw]
    proximity = measurements['proximity_draws'].mean(0) if draw is None else measurements['proximity_draws'][draw]
    if scores.shape[1:] != (len(periods), len(components)):
        raise ValueError('Period/component shape differs from the frozen specification')
    exposure = scores[local].reshape(len(local), dimension)
    raw = np.column_stack([recording, proximity.mean(2)])[local]
    nuisance = (raw - configuration['nuisance_center']) / configuration['nuisance_sd']
    controls = np.column_stack([np.ones(len(local)), cohort['controls'][local], nuisance])
    extra = np.empty((len(local), 0))
    if quadratic:
        xy = cohort['xy'][local] / 100000
        if quadratic_reference is None:
            mu = np.average(xy, axis=0, weights=weights)
            sd = np.sqrt(np.average((xy - mu)**2, axis=0, weights=weights))
        else:
            mu = np.asarray(quadratic_reference["coordinate_center"])
            sd = np.asarray(quadratic_reference["coordinate_sd"])
        z = (xy - mu) / sd
        rawq = np.column_stack([z[:, 1]**2, z[:, 1]*z[:, 0], z[:, 0]**2])
        if quadratic_reference is None:
            qm = np.average(rawq, axis=0, weights=weights)
            qs = np.sqrt(np.average((rawq - qm)**2, axis=0, weights=weights))
        else:
            qm = np.asarray(quadratic_reference["polynomial_center"])
            qs = np.asarray(quadratic_reference["polynomial_sd"])
        extra = (rawq - qm) / qs
    terms = [(period + '__' + component, [(i*len(components)+j, 1)])
             for i, period in enumerate(periods) for j, component in enumerate(components)]
    for earlier, later in configuration['period_differences']:
        for j, component in enumerate(components):
            terms.append((earlier + '_minus_' + later + '__' + component,
                          [(periods.index(earlier)*len(components)+j, 1),
                           (periods.index(later)*len(components)+j, -1)]))
    output = {}
    for origin, frozen in configuration['cells'].items():
        cells, index = np.unique(np.floor((cohort['xy'][local] - int(origin)) / configuration['cell_size_m']).astype(int), axis=0, return_inverse=True)
        if not np.array_equal(cells, frozen['coordinates']):
            raise ValueError('Regional membership differs from the frozen design')
        regional = helmert(len(cells), full=False).T[index]
        x = np.column_stack([controls, regional, extra, exposure - np.asarray(frozen['exposure_centers'])[index]])
        scales = [2] + [.5]*(controls.shape[1]-1) + [2]*(len(cells)-1) + [.5]*extra.shape[1] + [1]*dimension
        vectors = {}
        categories = len(configuration['categories'])
        for label, entries in terms:
            for name, contrast in configuration['lineage_contrasts'].items():
                contrast = np.asarray(contrast)
                if contrast.shape != (categories,) or not abs(contrast.sum()) < 1e-12:
                    raise ValueError('Lineage contrasts must sum to zero over all categories')
                vector = np.zeros((categories-1, x.shape[1]))
                for column, sign in entries:
                    vector[:, -dimension+column] = sign * contrast[1:]
                vectors[label + '__' + name] = vector.ravel()
        unit_xy = np.add.reduceat(cohort['xy'][local], cohort['starts'], axis=0) / counts[:, None]
        centroids = np.floor((unit_xy - int(origin)) / configuration['cell_size_m']).astype(int)
        output[origin] = dict(X=x, y=cohort['y'], counts=counts, scales=np.array(scales),
                              vectors=vectors, centroid_labels=np.array([str(a)+','+str(b) for a,b in centroids]),
                              regional_index=index, exposure_columns=dimension)
    return output

"""
BoTorch/GPyTorch multi-task GP for the gray-box h(x) output vector.

This model is optional. If torch/botorch/gpytorch are unavailable, callers
should fall back to the existing independent sklearn GPs.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class MultiTaskMagnitudeGP:
    """Thin wrapper around BoTorch MultiTaskGP for the magnitude vector h(x)."""

    def __init__(self, filters, feature_scaler=None):
        self.filters = list(filters)
        self.filter_to_task = {filt: i for i, filt in enumerate(self.filters)}
        self.feature_scaler = feature_scaler or StandardScaler()
        self.model = None
        self.likelihood = None
        self.available = False
        self.error_message = None

    def fit(self, X_train, y_train_mags, max_fit_iter=50):
        try:
            import torch
            import gpytorch
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import MultiTaskGP
            from botorch.models.transforms.outcome import Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except Exception as exc:
            self.error_message = str(exc)
            self.available = False
            return self

        X_scaled = self.feature_scaler.fit_transform(np.asarray(X_train, dtype=float))

        train_rows = []
        train_y = []
        for task_idx, filt in enumerate(self.filters):
            if filt not in y_train_mags:
                continue
            y = np.asarray(y_train_mags[filt], dtype=float)
            mask = np.isfinite(y)
            if not np.any(mask):
                continue

            task_col = np.full((int(mask.sum()), 1), float(task_idx))
            train_rows.append(np.hstack([X_scaled[mask], task_col]))
            train_y.append(y[mask].reshape(-1, 1))

        if not train_rows:
            self.error_message = "No valid magnitude training rows for MultiTaskGP."
            self.available = False
            return self

        train_X = torch.tensor(np.vstack(train_rows), dtype=torch.double)
        train_Y = torch.tensor(np.vstack(train_y), dtype=torch.double)

        self.model = MultiTaskGP(
            train_X,
            train_Y,
            task_feature=train_X.shape[-1] - 1,
            outcome_transform=Standardize(m=1),
        )
        self.likelihood = self.model.likelihood
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()
        with gpytorch.settings.max_cholesky_size(2000):
            fit_gpytorch_mll(mll, options={"maxiter": max_fit_iter})

        self.model.eval()
        self.likelihood.eval()
        self.available = True
        self.error_message = None
        return self

    def _make_task_test_X(self, X, filters=None):
        import torch

        if filters is None:
            filters = self.filters
        X = np.asarray(X, dtype=float)
        X = np.atleast_2d(X)
        X_scaled = self.feature_scaler.transform(X)

        rows = []
        row_filters = []
        for x_scaled in X_scaled:
            for filt in filters:
                task_idx = self.filter_to_task[filt]
                rows.append(np.r_[x_scaled, float(task_idx)])
                row_filters.append(filt)

        return torch.tensor(np.vstack(rows), dtype=torch.double), row_filters

    def posterior_h(self, X, filters=None):
        """
        Joint posterior for h(x) at one candidate.

        Returns:
            mean: shape (n_filters,)
            cov: shape (n_filters, n_filters)
            filters: list of filters in the covariance order
        """
        if not self.available or self.model is None:
            raise RuntimeError(self.error_message or "MultiTaskGP is not available.")

        import torch
        import gpytorch

        if filters is None:
            filters = self.filters
        filters = [f for f in filters if f in self.filter_to_task]

        test_X, _ = self._make_task_test_X(np.asarray(X, dtype=float).reshape(1, -1), filters)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model.posterior(test_X)

        mean = posterior.mean.detach().cpu().numpy().reshape(-1)
        cov = posterior.mvn.covariance_matrix.detach().cpu().numpy()
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        return mean, cov, filters

    def predict_batch(self, X, filters=None):
        """Posterior means/stds for each filter for a batch of candidates."""
        if filters is None:
            filters = self.filters
        filters = [f for f in filters if f in self.filter_to_task]

        magnitudes = {f: [] for f in filters}
        uncertainties = {f: [] for f in filters}
        for x in np.atleast_2d(X):
            mean, cov, ordered_filters = self.posterior_h(x, filters)
            std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
            for filt, mu, sig in zip(ordered_filters, mean, std):
                magnitudes[filt].append(mu)
                uncertainties[filt].append(sig)

        return (
            {f: np.asarray(v, dtype=float) for f, v in magnitudes.items()},
            {f: np.asarray(v, dtype=float) for f, v in uncertainties.items()},
        )


def build_multitask_magnitude_gp(X_train, y_train_mags, filters, max_fit_iter=50):
    model = MultiTaskMagnitudeGP(filters=filters)
    return model.fit(X_train, y_train_mags, max_fit_iter=max_fit_iter)

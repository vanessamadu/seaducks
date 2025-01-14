''' copied from ngboost.distns.multivariate_normal but removing the wrapper'''


import numpy as np
import scipy 
from ngboost.distns.distn import RegressionDistn
from ngboost.distns.multivariate_normal import get_chol_factor, MVNLogScore

class MVN(RegressionDistn):
        """
        Implementation of the Multivariate normal distribution for regression.
        Using the parameterization Sigma^{-1} = LL^T and modelling L
        diag(L) is exponentiated to ensure parameters are unconstrained.

        Scipy's multivariate normal benchmarks were relatively
        slow for pdf calculations so the implementation is from scratch.

        As Scipy has considerably more features call self.scipy_distribution()
        to return a list of distributions.
        """

        scores = [MVNLogScore]
        multi_output = True
        k = 2
        n_params = int(k * (k + 3) / 2)
        
        def __init__(self, params):
            super().__init__(params)
            self.N = params.shape[1]
            

            # Number of MVN dimensions, =k originally
            self.p = (-3 + (9 + 8 * self.n_params) ** 0.5) / 2
            self.p = int(self.p)

            # Extract parameters from params list
            # Param array is assumed to of shape n_params,N
            # First p rows are the mean
            # rest are the lower triangle of L.
            # Where Sigma_inverse = L@L.transpose()
            # Diagonals modelled on the log scale.
            self.loc = np.transpose(np.array(params[: self.p, :]))
            self.tril_L = np.array(params[self.p :, :])

            # Returns 3d array, shape (p, p, N)
            self.L = get_chol_factor(self.tril_L)

            # The remainder is just for utility.
            self.cov_inv = self.L @ self.L.transpose(0, 2, 1)

            # _cov_mat and _Lcov are place holders, relatively expensive to compute
            # The inverse of self.cov_inv. Access through self.cov
            self._cov_mat = None
            # cholesky factor of _cov_mat, useful for random number generation
            self._Lcov = None
            # Saving the pdf constant and means in an accessible format.
            self.pdf_constant = -self.p / 2 * np.log(2 * np.pi)

        def summaries(self, Y):
            """
            Parameters:
                Y: The data being fit to

            Returns:
                diff: N x p x1 the residual between the mean and the data
                eta: N x p which is L@diff
            """
            diff = np.expand_dims(self.loc - Y, 2)

            # N x 2 x 2 @ N x p x 1
            # -> N x p x 1 we remove the last index
            eta = np.squeeze(np.matmul(self.L.transpose(0, 2, 1), diff), axis=2)
            return diff, eta

        def logpdf(self, Y):
            _, eta = self.summaries(Y)
            # the exponentiated part of the pdf:
            p1 = -0.5 * np.sum(np.square(eta), axis=1)
            p1 = p1.squeeze()
            # this is the sqrt(determinant(Sigma)) component of the pdf
            p2 = np.sum(np.log(np.diagonal(self.L, axis1=1, axis2=2)), axis=1)

            ret = p1 + p2 + self.pdf_constant
            return ret

        def fit(Y):
            N, p = Y.shape
            m = Y.mean(axis=0)  # pylint: disable=unexpected-keyword-arg
            diff = Y - m
            sigma = 1 / N * (diff[:, :, None] @ diff[:, None, :]).sum(0)
            L = scipy.linalg.cholesky(np.linalg.inv(sigma), lower=True)
            diag_idx = np.diag_indices(p)
            L[diag_idx] = np.log(L[diag_idx])
            return np.concatenate([m, L[np.tril_indices(p)]])

        def rv(self):
            # This is only useful for generating rvs so only compute it in rv.
            if self._Lcov is None:
                self._Lcov = np.linalg.cholesky(np.linalg.inv(self.cov_inv))

            u = np.random.normal(loc=0, scale=1, size=(self.N, self.p, 1))
            sample = np.expand_dims(self.loc, 2) + self._Lcov @ u
            return np.squeeze(sample)

        def rvs(self, n):
            return [self.rv() for _ in range(n)]

        def sample(self, n):
            return self.rvs(n)

        @property
        def cov(self):
            # Covariance matrix is for computing the fisher information
            # If it is singular it is set an extremely large value.
            # This will probably not affect anything.
            if self._cov_mat is None:
                self._cov_mat = np.linalg.inv(self.cov_inv)
            return self._cov_mat

        @property
        def params(self):
            return {"loc": self.loc, "scale": self.cov}

        def scipy_distribution(self):
            """
            Returns:
                List of scipy.stats.multivariate_normal distributions.
            """
            cov_mat = self.cov
            return [
                scipy.stats.multivariate_normal(mean=self.loc[i, :], cov=cov_mat[i, :, :])
                for i in range(self.N)
            ]

        def mean(self):
            return self.loc
import sys
import os

import sklearn
from sklearn.linear_model import LogisticRegression


__all__ = [
    'Logistic'
]


class Logistic(LogisticRegression):
    """
    A custom Logistic Regression model that inherits from 
    sklearn.linear_model.LogisticRegression but overrides 
    the `predict` method to return class probabilities instead 
    of class labels.

    This subclass maintains full compatibility with the original 
    LogisticRegression class, including its initialization and 
    all methods.

    Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.

    pos_class : int, default=None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int or array-like of shape (n_cs,), default=10
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    fit_intercept : bool, default=True
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int, default=100
        Maximum number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, \
            default='lbfgs'
        Numerical solver to use.

    coef : array-like of shape (n_features,), default=None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.

    intercept_scaling : float, default=1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : {'ovr', 'multinomial', 'auto'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    check_input : bool, default=True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default=None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like of shape(n_samples,), default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    n_threads : int, default=1
       Number of OpenMP threads to use.

    Returns
    -------
    coefs : ndarray of shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array of shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """

    def __init__(self, penalty='l2', *, dual=False, tol=1e-4, C=1.0, fit_intercept=True, 
                 intercept_scaling=1, class_weight=None, random_state=None, solver='saga', 
                 max_iter=10000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, 
                 l1_ratio=None):
        super().__init__(
            penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, 
            intercept_scaling=intercept_scaling, class_weight=class_weight, 
            random_state=random_state, solver=solver, max_iter=max_iter, 
            multi_class=multi_class, verbose=verbose, warm_start=warm_start, 
            n_jobs=n_jobs, l1_ratio=l1_ratio)

    def predict(self, X):
        """
        This overrides the default `predict` method to return the same
        output as `predict_proba`.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        return super().predict_proba(X)[:,0]


















if __name__ == "__main__":
    ...
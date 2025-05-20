import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model

import sklearn.metrics


__all__ = [
    'GoodnessFit'
]


class GoodnessFit:
    """
    A collection of static methods for evaluating the goodness-of-fit, calibration,
    and discrimination of predictive models, especially for binary classification.

    This class provides tools to compute metrics such as:
    - R² and adjusted R²
    - AIC, BIC, and related model comparison scores
    - Log-likelihood, deviance, and calibration diagnostics
    - Statistical tests (Wald, LRT, Hosmer-Lemeshow, Rao Score)
    - Gini coefficient, AUC variance, and DeLong comparisons

    All methods are implemented as `@staticmethod` and designed for
    modular use in credit scoring, machine learning model validation, and
    statistical model assessment.

    Examples
    --------
    >>> GoodnessFit.r2_adjusted(y_true, y_pred, n_features=5)
    0.84

    >>> GoodnessFit.aic(y_true, y_pred, n_features=5)
    134.72
    """
    
    @staticmethod
    def ensure_prob_of_class_1(y_pred, prob_base_0=True):
        """
        Ensure that predicted probabilities correspond to the default value (class 1).

        This function is useful when model outputs are for class 0 by default,
        and need to be flipped to match class 1 probabilities for consistency
        in scoring and evaluation.

        Parameters
        ----------
        y_pred : array-like
            Predicted probabilities, either for class 0 or class 1.
        prob_base_0 : bool, optional
            If True (default), assumes the probabilities are for class 0
            and returns `1 - y_pred`. If False, assumes probabilities are already
            for class 1 and returns them unchanged.

        Returns
        -------
        np.ndarray
            Array of probabilities for the positive class (class 1).
        """
        y_pred = np.array(y_pred)
        return 1 - y_pred if prob_base_0 else y_pred


    @staticmethod
    def r2(y_true:list, y_pred:list, **kwargs):
        """
        Compute the R² (coefficient of determination) regression score.

        Parameters
        ----------
        y_true : list
            Ground truth (correct) target values.
        y_pred : list
            Estimated target values.
        **kwargs : dict
            Additional arguments passed to `sklearn.metrics.r2_score`.

        Returns
        -------
        float
            R² score, rounded to 4 decimal places.
        """
        value = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred, **kwargs)
        return round(value,4)
    

    @staticmethod
    def r2_adjusted(y_true:list, y_pred:list, n_features:int, **kwargs):
        """
        Compute the adjusted R² (coefficient of determination) score.

        Adjusted R² accounts for the number of predictors in the model,
        penalizing models with more features unless they improve performance.

        Parameters
        ----------
        y_true : list
            Actual target values.
        y_pred : list
            Predicted target values.
        n_features : int
            Number of features (independent variables) used in the model.
        **kwargs : dict
            Additional keyword arguments passed to `sklearn.metrics.r2_score`.

        Returns
        -------
        float
            Adjusted R² score.

        Raises
        ------
        ZeroDivisionError
            If `n_features` is greater than or equal to `len(y_true) - 1`.
        """
        n = len(y_true)
        if (n <= n_features+1):
            raise ZeroDivisionError(f'"n_features" must be less than {len(y_true)-1}')
        r2 = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred, **kwargs)
        value = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return value
    

    @staticmethod
    def chi2(observed:list, expected:list, alpha:float=0.05, info:bool=False):
        """
        Perform a Chi-squared goodness-of-fit test.

        This test compares observed frequencies with expected frequencies
        to evaluate whether they differ significantly.

        Parameters
        ----------
        observed : list
            Observed frequencies.
        expected : list
            Expected frequencies under the null hypothesis.
        alpha : float, optional
            Significance level for hypothesis testing (default is 0.05).
        info : bool, optional
            If True, return detailed test output including p-value and decision.

        Returns
        -------
        float or dict
            If `info` is False: the Chi-squared statistic.
            If `info` is True: dictionary with keys:
                - "chi2": Chi-squared statistic
                - "p value": p-value of the test
                - "critical value": critical value at the given alpha
                - "reject null": whether the null hypothesis is rejected
                - "conclusion": interpretation of the result
        """
        observed = np.array(observed)
        expected = np.array(expected)
        chi2_statistic = np.sum((observed - expected)**2 / expected)
        df = len(observed) - 1
        critical_value = scipy.stats.chi2.ppf(1 - alpha, df)
        p_value_ = 1 - scipy.stats.chi2.cdf(chi2_statistic, df)
        reject_null = bool(chi2_statistic > critical_value)

        if (reject_null == True):
            conclusion = 'Failed to Follow Expected Distribution'
        else:
            conclusion = 'Followed Expected Distribution'

        if (info == True):
            return {
                "chi2": float(round(chi2_statistic,4)),
                "p value": float(p_value_),
                "critical value": float(critical_value),
                "reject null": reject_null,
                'conclusion': conclusion
            }

        return float(round(chi2_statistic,4))


    @staticmethod
    def hosmer_lemeshow(y_true:list, y_pred:list, g:int=10, info:bool=False, prob_base_0:bool=True):
        """
        Perform the Hosmer-Lemeshow goodness-of-fit test for logistic regression models.

        The test compares observed and predicted event rates across quantile-based groups
        to evaluate model calibration.

        Parameters
        ----------
        y_true : list
            Binary ground truth labels (0 or 1).
        y_pred : list
            Predicted probabilities from a binary classifier.
        g : int, optional
            Number of quantile bins to group predictions (default is 10).
        info : bool, optional
            If True, return a dictionary with full test results (default is False).
        prob_base_0 : bool, optional
            If True, assumes `y_pred` is probability of class 0 and converts to class 1 (default is True).

        Returns
        -------
        float or dict
            If `info` is False: HL test statistic.
            If `info` is True: dictionary with:
                - 'HL': HL statistic
                - 'p value': p-value from Chi-squared distribution
                - 'degrees of freedom': g - 2
                - 'reject null': whether null hypothesis is rejected
                - 'conclusion': textual interpretation

        Notes
        -----
        The null hypothesis assumes that predicted probabilities fit the observed outcomes well.
        """
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)

        data = pd.DataFrame({'y': y_true, 'p': y_pred})
        data['group'] = pd.qcut(data['p'], q=g, duplicates='drop')

        grouped = data.groupby('group', observed=False)
        hl_statistic = 0
        for _, group in grouped:
            obs = group['y'].sum()
            exp = group['p'].sum()
            n = len(group)
            p_hat = exp / n
            if p_hat in [0, 1]:  # evita divisão por zero
                continue
            hl_statistic += ((obs - exp) ** 2) / (n * p_hat * (1 - p_hat))

        df = g - 2
        p_value_ = 1 - scipy.stats.chi2.cdf(hl_statistic, df)
        reject_null = p_value_ < 0.05
        if (reject_null == True):
            conclusion = 'Not Well Ajusted'
        else:
            conclusion = 'Well Ajusted'

        if (info == True):
            return {
                'HL': float(round(hl_statistic,4)),
                'p value': float(p_value_),
                'degrees of freedom': df,
                'reject null': bool(reject_null),
                'conclusion': conclusion
            }

        return round(hl_statistic,4)


    @staticmethod
    def log_likelihood(y_true:list, y_pred:list, return_individual:bool=False):
        """
        Compute the log-likelihood of a binary classification model.

        Parameters
        ----------
        y_true : list
            Binary ground truth labels (0 or 1).
        y_pred : list
            Predicted probabilities for class 1.
        return_individual : bool, optional
            If True, return the individual log-likelihood values (default is False).

        Returns
        -------
        float or np.ndarray
            Total log-likelihood (if `return_individual` is False),
            or array of individual log-likelihoods (if True).

        Notes
        -----
        The log-likelihood is computed as:
            logL = sum(y_true * log(p) + (1 - y_true) * log(1 - p))
        where `p` is the predicted probability of class 1.
        """
        y_true = np.array(y_true)
        y_pred = np.clip(np.array(y_pred), 1e-10, 1 - 1e-10)
        
        log_likelihood = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        value = np.sum(log_likelihood)

        if (return_individual == True):
            return log_likelihood

        return float(round(value,4))


    @staticmethod
    def deviance(y_true:list, y_pred:list, return_individual:bool=False, prob_base_0:bool=True):
        """
        Compute the deviance for a binary classification model.

        Deviance is a measure of model fit based on the log-likelihood; lower values indicate better fit.

        Parameters
        ----------
        y_true : list
            Binary ground truth labels (0 or 1).
        y_pred : list
            Predicted probabilities for class 0 or 1.
        return_individual : bool, optional
            If True, return individual deviance residuals instead of the sum (default is False).
        prob_base_0 : bool, optional
            If True, assumes `y_pred` is probability of class 0 and converts to class 1 (default is True).

        Returns
        -------
        float or np.ndarray
            Total deviance (if `return_individual` is False),
            or array of individual deviance residuals (if True).

        Notes
        -----
        Deviance is defined as:
            D = -2 * log-likelihood
        and residuals are:
            sqrt(2 * -log-likelihood)
        """
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)

        log_likelihoods = GoodnessFit.log_likelihood(y_true, y_pred, return_individual=True)
        deviance_residuals = np.sqrt(2 * -log_likelihoods)
        value = np.sum(deviance_residuals ** 2)

        if (return_individual == True):
            return deviance_residuals

        return float(round(value,4))


    @staticmethod
    def aic(y_true:list, y_pred:list, n_features:int, prob_base_0:bool=True):
        """
        Compute the Akaike Information Criterion (AIC) for model evaluation.

        AIC balances model fit and complexity by penalizing models with more parameters.
        Lower AIC values indicate a better model.

        Parameters
        ----------
        y_true : list
            Binary ground truth labels (0 or 1).
        y_pred : list
            Predicted probabilities for class 0 or 1.
        n_features : int
            Number of independent features used in the model.
        prob_base_0 : bool, optional
            If True, converts probabilities from class 0 to class 1 (default is True).

        Returns
        -------
        float
            AIC value, rounded to 4 decimal places.

        Notes
        -----
        AIC is computed as:
            AIC = 2 * k - 2 * logL
        where `k` is the number of parameters and `logL` is the log-likelihood.
        """
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)
        log_likelihood = GoodnessFit.log_likelihood(y_true, y_pred)
        value = 2 * n_features - 2 * log_likelihood
        return float(round(value,4))


    @staticmethod
    def aic_small_sample(y_true:list, y_pred:list, n_features:int, sample_size:int, prob_base_0:bool=True):
        """
        Compute the corrected AIC (AICc) for small sample sizes.

        This correction adjusts AIC to account for bias when the number of observations is
        small relative to the number of model parameters.

        Parameters
        ----------
        y_true : list
            Binary ground truth labels (0 or 1).
        y_pred : list
            Predicted probabilities for class 0 or 1.
        n_features : int
            Number of features (parameters) used in the model.
        sample_size : int
            Total number of observations.
        prob_base_0 : bool, optional
            If True, converts probabilities from class 0 to class 1 (default is True).

        Returns
        -------
        float
            Corrected AIC (AICc), rounded to 4 decimal places.

        Raises
        ------
        ValueError
            If `sample_size` is less than or equal to `n_features + 1`.

        Notes
        -----
        AICc is computed as:
            AICc = AIC + [2k(k + 1)] / (n - k - 1)
        where `k` is number of features and `n` is sample size.
        """
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)
        aic = GoodnessFit.aic(y_true, y_pred, n_features=n_features)
        if sample_size <= n_features + 1:
            raise ValueError(f'"Sample Size" must be at least {n_features+1}')
        value = aic + (2 * n_features * (n_features + 1)) / (sample_size - n_features - 1)
        return float(round(value,4))


    @staticmethod
    def relative_likelihood(aic_values: list) -> list:
        """
        Compute the relative likelihoods of models based on AIC values.

        The relative likelihood measures how probable each model is
        compared to the best model (lowest AIC).

        Parameters
        ----------
        aic_values : list
            AIC scores from different models.

        Returns
        -------
        list
            List of relative likelihoods, rounded to 4 decimal places.

        Notes
        -----
        Relative likelihood is computed as:
            exp((AIC_min - AIC_i) / 2)
        where AIC_min is the lowest AIC value in the list.
        """
        aic_min = np.min(aic_values)
        values = np.exp((aic_min - aic_values) / 2)
        rounded_values = [round(val, 4) for val in values]
        return rounded_values


    @staticmethod
    def weighted_average_estimate(aic_values:list, estimates:list):
        """
        Compute a weighted average of estimates based on AIC-derived relative likelihoods.

        Useful for model averaging, where predictions or coefficients from multiple models
        are combined based on their relative support.

        Parameters
        ----------
        aic_values : list
            AIC values corresponding to each model.
        estimates : list
            A list of estimates (e.g., predictions or coefficients) from each model.

        Returns
        -------
        float
            Weighted average estimate, rounded to 4 decimal places.

        Notes
        -----
        The weights are derived using:
            weight_i = exp((AIC_min - AIC_i) / 2)
        Then a weighted average is computed over the estimates.
        """
        relative_likelihoods = GoodnessFit.relative_likelihood(aic_values=aic_values)
        estimates = np.array(estimates)
        relative_likelihoods = np.array(relative_likelihoods)
        value = np.sum(estimates * relative_likelihoods) / np.sum(relative_likelihoods)
        return float(round(value,4))
    

    @staticmethod
    def weighted_estimates_batch(aic_values:list, predictions_per_model:list[list]) -> list:
        """
        Compute weighted average predictions for multiple observations using AIC-based weights.

        Each observation has predictions from multiple models. This method applies model
        averaging across those models using AIC-based relative likelihoods.

        Parameters
        ----------
        aic_values : list
            AIC values for each model.
        predictions_per_model : list of lists
            Each inner list contains predictions for all observations from one model.
            Outer list size: number of models.
            Inner list size: number of observations.

        Returns
        -------
        list
            Weighted average predictions per observation, one for each data point.

        Example
        -------
        >>> GoodnessFit.weighted_estimates_batch(
        ...     aic_values=[120.5, 118.0, 122.2],
        ...     predictions_per_model=[[0.7, 0.6], [0.75, 0.65], [0.72, 0.62]]
        ... )
        [0.74, 0.64]
        """
        predictions_per_person = list(zip(*predictions_per_model))

        weighted_predictions = [
            GoodnessFit.weighted_average_estimate(aic_values, person_preds)
            for person_preds in predictions_per_person
        ]

        return weighted_predictions
    

    @staticmethod
    def bic(y_true:list, y_pred:list, n_features:int, sample_size:int, prob_base_0:bool=True):
        """
        Compute the Bayesian Information Criterion (BIC) for model evaluation.

        BIC penalizes model complexity more strongly than AIC. It is commonly used
        for model selection when sample size is relatively large.

        Parameters
        ----------
        y_true : list
            Binary ground truth labels (0 or 1).
        y_pred : list
            Predicted probabilities for class 0 or 1.
        n_features : int
            Number of independent features used in the model.
        sample_size : int
            Total number of observations.
        prob_base_0 : bool, optional
            If True, converts probabilities from class 0 to class 1 (default is True).

        Returns
        -------
        float
            BIC value, rounded to 4 decimal places.

        Notes
        -----
        BIC is computed as:
            BIC = log(n) * k - 2 * logL
        where:
            - `n` is sample size
            - `k` is number of features
            - `logL` is the log-likelihood of the model
        """
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)
        ll = GoodnessFit.log_likelihood(y_true, y_pred)
        value = np.log(sample_size) * n_features - 2 * ll
        return float(round(value, 4))


    @staticmethod
    def likelihood_ratio_test(loglike_simple:float, loglike_complex:float, added_params:int, alpha:float=0.05, info:bool=False):
        """
        Perform a likelihood ratio test (LRT) to compare nested models.

        This test evaluates whether a more complex model provides a significantly better fit
        than a simpler (nested) model by comparing their log-likelihoods.

        Parameters
        ----------
        loglike_simple : float
            Log-likelihood of the simpler (null) model.
        loglike_complex : float
            Log-likelihood of the more complex (alternative) model.
        added_params : int
            Number of additional parameters in the complex model.
        alpha : float, optional
            Significance level for hypothesis testing (default is 0.05).
        info : bool, optional
            If True, return a dictionary with full test results.

        Returns
        -------
        float or dict
            If `info` is False: LRT statistic.
            If `info` is True: dictionary with keys:
                - "LRT statistic"
                - "p value"
                - "reject null"
                - "conclusion"

        Notes
        -----
        The test statistic is:
            LRT = -2 * (loglike_simple - loglike_complex)
        which follows a Chi-squared distribution with `added_params` degrees of freedom.
        """
        value = -2 * (loglike_simple - loglike_complex)
        p_value_ = 1 - scipy.stats.chi2.cdf(value, added_params)
        reject_null = bool(p_value_ < alpha)
        if (reject_null == True):
            conclusion = 'Statistical Improvement'
        else:
            conclusion = 'No Statistical Improvement'

        if (info == True):
            return {
                "LRT statistic": round(value,4),
                "p value": round(p_value_,4),
                "reject null": reject_null,
                'conclusion': conclusion
            }

        return float(round(value,4))
    

    @staticmethod
    def wald_test(y_true:list, y_pred:list, degrees_freedom:int=1, alpha:float=0.05, null_value:float=0, info:bool=False):
        """
        Perform a Wald test to evaluate whether a model coefficient is significantly different from a null value.

        This implementation uses logistic regression, where the model is fitted using predicted probabilities
        as the single explanatory variable.

        Parameters
        ----------
        y_true : list
            Binary ground truth labels (0 or 1).
        y_pred : list
            Predicted probabilities for class 1.
        degrees_freedom : int, optional
            Degrees of freedom for the Chi-squared distribution (default is 1).
        alpha : float, optional
            Significance level for hypothesis testing (default is 0.05).
        null_value : float, optional
            The null hypothesis value of the coefficient (default is 0).
        info : bool, optional
            If True, return a dictionary with full test results.

        Returns
        -------
        float or dict
            If `info` is False: Wald statistic.
            If `info` is True: dictionary with keys:
                - "Wald statistic"
                - "p value"
                - "reject null"
                - "conclusion"

        Notes
        -----
        The test statistic is:
            ((beta - null_value) / std_error)^2
        where `beta` is the estimated coefficient and `std_error` its standard error.
        """
        y_pred_noisy = np.array(y_pred) + np.random.normal(0, 1e-4, size=len(y_pred))
        
        X = np.array(y_pred_noisy).reshape(-1, 1)
        y = np.array(y_true)

        model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(X, y)

        beta = model.coef_[0][0]

        p = model.predict_proba(X)[:, 1]
        X_design = np.hstack([np.ones((len(X), 1)), X])
        V = np.diag(p * (1 - p))
        cov_matrix = np.linalg.inv(X_design.T @ V @ X_design)
        std_error = np.sqrt(cov_matrix[1, 1])  # std error for the score coefficient

        value = ((beta - null_value) / std_error) ** 2
        p_value_ = 1 - scipy.stats.chi2.cdf(value, degrees_freedom)
        reject_null = p_value_ < alpha
        conclusion = 'Informative Variable' if reject_null else 'Not Informative Variable'

        if info:
            return {
                "Wald statistic": round(value, 4),
                "p value": round(p_value_, 4),
                "reject null": bool(reject_null),
                'conclusion': conclusion
            }

        return float(round(value, 4))
    

    @staticmethod
    def wald_test_(beta:float=None, std_error:float=None, degrees_freedom:int=1, alpha:float=0.05, null_value:float=0, info:bool=False):
        """
        Compute the Wald test statistic given coefficient and standard error.

        This version is useful when coefficient estimates and their standard errors are
        already computed outside this method (e.g., from statsmodels or sklearn).

        Parameters
        ----------
        beta : float
            Estimated coefficient for the predictor.
        std_error : float
            Standard error of the estimated coefficient.
        degrees_freedom : int, optional
            Degrees of freedom for the Chi-squared distribution (default is 1).
        alpha : float, optional
            Significance level for hypothesis testing (default is 0.05).
        null_value : float, optional
            The null hypothesis value of the coefficient (default is 0).
        info : bool, optional
            If True, return a dictionary with full test results.

        Returns
        -------
        float or dict
            If `info` is False: Wald statistic.
            If `info` is True: dictionary with keys:
                - "Wald statistic"
                - "p value"
                - "reject null"
                - "conclusion"

        Notes
        -----
        Wald statistic is computed as:
            ((beta - null_value) / std_error)^2
        and follows a Chi-squared distribution with specified degrees of freedom.
        """
        value = ((beta - null_value) / std_error) ** 2
        p_value_ = 1 - scipy.stats.chi2.cdf(value, degrees_freedom)
        reject_null = bool(p_value_ < alpha)
        if (reject_null == True):
            conclusion = 'Informative Variable'
        else:
            conclusion = 'Not Informative Variable'

        if (info == True):
            return {
                "Wald statistic": round(value,4),
                "p value": round(p_value_,4),
                "reject null": reject_null,
                'conclusion': conclusion
            }

        return float(round(value,4))
    

    @staticmethod
    def rao_score_test(y_pred:pd.DataFrame, y_true:pd.Series, X_candidate:pd.Series, 
                       family:str='normal', phi:float=None, alpha:float=0.05, info:bool=False):
        """
        Perform the Rao (Score) test to assess the contribution of a candidate predictor.

        The Rao score test evaluates whether a new predictor variable significantly improves
        the fit of a generalized linear model (GLM) without refitting the model.

        Parameters
        ----------
        y_pred : pd.DataFrame
            Predicted values (e.g., means) from the current model.
        y_true : pd.Series
            Actual target values.
        X_candidate : pd.Series
            Values of the candidate predictor variable.
        family : str, optional
            Distribution family of the GLM (e.g., "binomial", "poisson", "normal").
        phi : float, optional
            Dispersion parameter (used in quasi-families and Gaussian models).
        alpha : float, optional
            Significance level for hypothesis testing (default is 0.05).
        info : bool, optional
            If True, return detailed output including p-value and interpretation.

        Returns
        -------
        float or dict
            If `info` is False: Rao score Chi² statistic.
            If `info` is True: dictionary with:
                - "Score_Chi2"
                - "p_value"
                - "reject_null"
                - "conclusion"

        Raises
        ------
        ValueError
            If the distribution family is not recognized.

        Notes
        -----
        This test avoids refitting by using the score function and Fisher information
        evaluated at the restricted (null) model.
        """
        family = family.lower()
        if ('binom' in family):
            var = y_pred * (1 - y_pred)
        elif ('quasibinom' in family):
            var = phi * y_pred * (1 - y_pred)
        elif ('poisson' in family):
            var = y_pred
        elif ('quasipoisson' in family):
            var = phi * y_pred
        elif ('gauss' in family) or ('normal' in family):
            var = np.full_like(y_pred, phi)
        elif ('gamma' in family):
            var = phi * y_pred**2
        elif ('inv' in family) and ('gauss' in family):
            var = phi * y_pred**3
        elif ('neg' in family) and ('binom' in family):
            var = y_pred + alpha * y_pred**2
        else:
            raise ValueError(f"Família '{family}' não reconhecida.")

        score = np.sum(X_candidate * (y_true - y_pred))
        fisher_info = np.sum(X_candidate ** 2 * var)

        value = (score ** 2) / fisher_info
        p_value_ = 1 - scipy.stats.chi2.cdf(value, df=1)
        reject_null = bool(p_value_ < alpha)
        if (reject_null == True):
            conclusion = 'Variable Significantly Improve the Model'
        else:
            conclusion = 'Variable Do Not Improve the Model'

        if (info == True):
            return {
                "Score_Chi2": float(round(value,4)),
                "p_value": float(round(p_value_,4)),
                "reject_null": reject_null,
                'conclusion': conclusion
            }

        return float(round(value,4))
    

    @staticmethod
    def deviance_odds(y_true:list, y_pred:list, final_value:bool=True, info:bool=False, prob_base_0:bool=True):
        """
        Evaluate model quality using deviance-based odds metrics: predictive power and naïve accuracy.

        This diagnostic summarizes how well the predicted probabilities rank and calibrate against
        observed binary outcomes.

        Parameters
        ----------
        y_true : list
            Binary ground truth labels (0 or 1).
        y_pred : list
            Predicted probabilities for class 0 or 1.
        final_value : bool, optional
            If True, return just `(power, accuracy)` tuple. If False, return full DataFrame.
        info : bool, optional
            If True, return detailed metrics and textual conclusion.
        prob_base_0 : bool, optional
            If True, assumes predictions are for class 0 and converts to class 1.

        Returns
        -------
        tuple, dict, or pd.DataFrame
            - If `final_value` is True: (power, accuracy) tuple.
            - If `info` is True: dict with keys: 'power', 'accuracy', 'conclusion'.
            - If `final_value` is False and `info` is False: DataFrame with internal calculations.

        Notes
        -----
        - Predictive power estimates the model's ability to rank cases.
        - Naïve accuracy estimates alignment of predicted probabilities with outcome frequencies.
        - Based on deviance and expected likelihood.
        """
        assert set(np.unique(y_true)).issubset({0, 1})
        assert (y_pred.min() >= 0) and (y_pred.min() <= 1)
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)

        dff = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
        })

        dff['l'] = GoodnessFit.log_likelihood(dff['y_true'], dff['y_pred'], return_individual=True)
        dff['D'] = -2 * dff['l']
        D_bar = dff['D'].sum() / len(dff)
        psi_b = np.exp(D_bar)

        dff['mu_o'] = dff['y_true'].mean()
        dff['l_o'] = dff['y_true']*np.log(dff['mu_o']) + (1-dff['y_true'])*np.log(1-dff['mu_o'])
        dff['D_o'] = -2*dff['l_o']
        D_bar_o = dff['D_o'].sum()/len(dff)
        psi_o = np.exp(D_bar_o)

        dff['mu_b'] = dff['y_pred'].mean()
        dff['l_b'] = dff['y_true']*np.log(dff['mu_b']) + (1-dff['y_true'])*np.log(1-dff['mu_b'])
        dff['D_b'] = -2*dff['l_b']
        D_bar_b = dff['D_b'].sum()/len(dff)
        psi_e = np.exp(D_bar_b)

        power = float(round(100* (psi_e-psi_b)/(psi_e-1),2))
        accuracy = float(round(100* (1-(psi_e-psi_o)/psi_e),2))

        dff.loc['Total',:] = dff.sum(axis=0)

        conclusion = ''
        # Power interpretation
        if power < 0:
            conclusion += "<!> The model has negative predictive power, meaning it ranks outcomes worse than random. This suggests either a serious model flaw or a reversal in prediction logic (e.g., predicting the opposite class). "
        elif power < 50:
            conclusion += "<!> The model has weak predictive power, indicating limited ability to rank or discriminate between outcomes. It may need retraining or feature engineering. "
        elif power < 70:
            conclusion += "The model has moderate predictive power. It performs reasonably but could benefit from improvements. "
        elif power < 90:
            conclusion += "<o> The model has strong predictive power, suggesting effective ranking of predictions. "
        else:
            conclusion += "<o> The model has excellent predictive power, showing it ranks outcomes very effectively. "

        # Accuracy interpretation
        if accuracy < 0:
            conclusion += "<x> The model has negative naïve accuracy, which is highly problematic. Its probability estimates are worse than random — likely due to severe miscalibration or label errors. "
        elif accuracy < 70:
            conclusion += "<!> The model has poor naïve accuracy. Estimated probabilities deviate significantly from observed outcomes. Calibration is recommended. "
        elif accuracy < 90:
            conclusion += "The model has acceptable naïve accuracy, though some calibration error exists. "
        elif accuracy <= 100:
            conclusion += "<o> The model is well-calibrated, with high naïve accuracy suggesting predicted probabilities align closely with observed outcomes. "
        else:
            conclusion += "<!> Accuracy exceeds 100%, which may indicate a computation error. "
        conclusion = conclusion.strip()

        if (info == True):
            return {
                'power': power,
                'accuracy': accuracy,
                'conclusion': conclusion,
            }

        if (final_value == True):
            return (power, accuracy)

        return dff


    @staticmethod
    def calinski_harabasz(y_pred:list, bins:list):
        """
        Compute the Calinski-Harabasz score to evaluate the separation between binned groups.

        This score measures how well predicted probabilities are clustered by a given binning scheme.
        It is commonly used in clustering and binning evaluations to assess group distinctiveness.

        Parameters
        ----------
        y_pred : list
            Predicted probabilities or scores (continuous values).
        bins : list
            Corresponding bin labels for each prediction (can be strings or numeric identifiers).

        Returns
        -------
        float
            Calinski-Harabasz score, rounded to 4 decimal places.
            Returns `np.inf` if the denominator is zero (suggesting perfect separation).
            Returns `0` if the result is not a number (`NaN`).

        Notes
        -----
        The score is calculated as:

            CH = (BSS / (g - 1)) / (WSS / (n - g))

        where:
        - BSS = between-group sum of squares
        - WSS = within-group sum of squares
        - g = number of bins
        - n = total number of observations

        A higher score indicates better separation between the groups.
        """
        df = {
            "y_pred": y_pred,
            "bins": bins
        }
        df = pd.DataFrame(df)
        
        overall_mean = df['y_pred'].mean()
        n = len(df)
        g = df['bins'].nunique()

        bss = (
            df.groupby('bins')['y_pred']
            .apply(lambda x: len(x) * (x.mean() - overall_mean) ** 2)
            .sum()
        )

        wss = (
            df.groupby('bins')['y_pred']
            .apply(lambda x: ((x - x.mean()) ** 2).sum())
            .sum()
        )

        if ((wss / (n - g)) == 0):
            return np.inf

        ch = (bss / (g - 1)) / (wss / (n - g))

        if np.isnan(ch):
            return 0
        
        return float(round(ch,4))


    @staticmethod
    def gini_variance(y_true:list, y_pred:list, info:bool=False, prob_base_0:bool=True, **kwargs):
        """
        Estimate the variance and confidence intervals of the Gini coefficient using multiple methods.

        This function computes AUC and Gini, and estimates variance using the Van Dantzig,
        Bamber, and Engelmann approaches. Engelmann’s method is used to derive a 95% CI for Gini.

        Parameters
        ----------
        y_true : list
            Binary ground truth labels (0 or 1).
        y_pred : list
            Predicted probabilities for class 0 or class 1.
        info : bool, optional
            If True, returns a dictionary with detailed statistics (default is False).
        prob_base_0 : bool, optional
            If True, assumes `y_pred` is the probability for class 0 and converts it to class 1.
        **kwargs : dict
            Additional keyword arguments passed to `sklearn.metrics.roc_auc_score`.

        Returns
        -------
        float or dict
            If `info` is False: variance of Gini (Engelmann method).
            If `info` is True: dictionary with keys:
                - "AUC", "Gini"
                - "N_G", "N_B"
                - "Var (Van Dantzig)", "Var (Bamber)", "Var (Engelmann)"
                - "Gini CI Lower", "Gini CI Upper"

        Notes
        -----
        Gini = 2 * AUC - 1.
        This method also estimates variance and confidence intervals of Gini using ranking-based statistics.
        """
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)
        dff = pd.DataFrame({'y_true':y_true, 'y_score':y_pred})
        dff = dff.sort_values(by='y_score', ascending=False).reset_index(drop=True)

        auc = sklearn.metrics.roc_auc_score(y_true, y_pred, **kwargs)
        gini = 2 * auc - 1

        N_B = np.sum(y_true) 
        N_G = np.sum(1 - y_true) 

        van_dantzig = (1 - gini**2) / min(N_G, N_B)
        bamber = ((2 * N_G + 1) * (1 - gini**2) - (N_G - N_B) * (1 - gini)**2) / (3 * N_G * N_B)

        dff['P(B)'] = dff['y_true'] / N_B 
        dff['P(G)'] = (1 - dff['y_true']) / N_G 

        dff['F(B)'] = dff['P(B)'].cumsum() 
        dff['F(G)'] = dff['P(G)'].cumsum() 

        term1 = (N_B - 1) * np.sum(dff['P(G)'] * (1 - 2 * dff['F(B)']) ** 2)
        term2 = (N_G - 1) * np.sum(dff['P(B)'] * (1 - 2 * dff['F(G)']) ** 2)
        term3 = -2 * np.sum(dff['P(G)'] * dff['P(B)'])
        term4 = -4 * (N_G + N_B - 1) * gini ** 2
        numerator = term1 + term2 + term3 + term4 + 1
    
        denominator = (N_G - 1) * (N_B - 1)
        engelmann = numerator / denominator

        gini_lower = gini - 1.96 * np.sqrt(engelmann)
        gini_upper = gini + 1.96 * np.sqrt(engelmann)

        if (info == True):
            return {
                "AUC": float(round(auc,4)),
                "Gini": float(round(gini,4)),
                "N_G": int(round(N_G,4)),
                "N_B": int(round(N_B,4)),
                "Var (Van Dantzig)": float(round(van_dantzig,4)),
                "Var (Bamber)": float(round(bamber,4)),
                "Var (Engelmann)": float(round(engelmann,4)),
                "Gini CI Lower": float(round(gini_lower,4)),
                "Gini CI Upper": float(round(gini_upper,4))
            }
        
        return float(round(engelmann,4))


    @staticmethod
    def delong_roc_variance(y_true:list, y_pred:list, prob_base_0:bool=True):
        """
        Estimate the variance of the AUC using the DeLong method.

        The DeLong method is a nonparametric approach to compute the standard error of
        the area under the ROC curve (AUC), taking into account ranking uncertainty.

        Parameters
        ----------
        y_true : list
            Binary ground truth labels (0 or 1).
        y_pred : list
            Predicted probabilities for class 1.
        prob_base_0 : bool, optional
            If True, assumes `y_pred` is the probability for class 0 and converts it to class 1.

        Returns
        -------
        tuple
            AUC and its variance, both rounded to 4 decimal places.

        Notes
        -----
        This function assumes independence between positive and negative classes.
        """
        if (prob_base_0 == True):
            y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred=y_pred)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        order = np.argsort(-y_pred)
        y_pred = y_pred[order]
        y_true = y_true[order]

        pos_count = np.sum(y_true)
        neg_count = len(y_true) - pos_count

        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]

        V10 = np.array([np.sum(score > neg_scores) + 0.5 * np.sum(score == neg_scores) for score in pos_scores]) / neg_count
        V01 = np.array([np.sum(score < pos_scores) + 0.5 * np.sum(score == pos_scores) for score in neg_scores]) / pos_count

        auc = np.mean(V10)
        s10 = np.var(V10, ddof=1) / pos_count
        s01 = np.var(V01, ddof=1) / neg_count

        auc_variance = s10 + s01
        return float(round(auc,4)), float(round(auc_variance,4))


    @staticmethod
    def compare_auc_delong(y_true:list, y_preds:list[list]):
        """
        Perform pairwise comparisons of AUCs using the DeLong test.

        This method compares the AUCs of all pairs of models and tests whether their
        differences are statistically significant using the DeLong variance estimates.

        Parameters
        ----------
        y_true : list
            Binary ground truth labels (0 or 1).
        y_preds : list of list
            Each inner list contains predicted probabilities from one model.
            Outer list length = number of models.
            Inner list length = number of observations.

        Returns
        -------
        list of dict
            Each dictionary contains:
                - 'model_1': index of first model
                - 'model_2': index of second model
                - 'auc1': AUC of first model
                - 'auc2': AUC of second model
                - 'z_stat': Z-score for the AUC difference
                - 'p_value': two-tailed p-value

        Notes
        -----
        Assumes model predictions are independent.
        """
        results = []
        n = len(y_preds)
        for i in range(n):
            for j in range(i + 1, n):
                auc1, var1 = GoodnessFit.delong_roc_variance(y_true, y_preds[i])
                auc2, var2 = GoodnessFit.delong_roc_variance(y_true, y_preds[j])

                auc_diff = auc1 - auc2
                var_diff = var1 + var2  # assumes independence
                z = auc_diff / np.sqrt(var_diff)
                p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z)))

                results.append({
                    'model_1': i,
                    'model_2': j,
                    'auc1': auc1,
                    'auc2': auc2,
                    'z_stat': z,
                    'p_value': p_value
                })
        return results


if __name__ == '__main__':
    df = {
        'Grade': [0]*(95+309) + [1]*(187+224) + [2]*(549+299) + [3]*(1409+495) + [4]*(3743+690) + [5]*(4390+424) + [6]*(2008+94) + [7]*(593+8),
        'Y': [0]*95+[1]*309 + [0]*187+[1]*224 + [0]*549+[1]*299 + [0]*1409+[1]*495 + [0]*3743+[1]*690 + [0]*4390+[1]*424 + [0]*2008+[1]*94 + [0]*593+[1]*8,
        'mu': [309/(95+309)]*(95+309) + [224/(187+224)]*(187+224) + [299/(549+299)]*(549+299) + [495/(1409+495)]*(1409+495) + [690/(3743+690)]*(3743+690) + [424/(4390+424)]*(4390+424) + [94/(2008+94)]*(2008+94) + [8/(593+8)]*(593+8)
    }
    df = pd.DataFrame(df)
    print(
        GoodnessFit.wald_test([1, 0, 1, 1, 0, 0], [0.9, 0.1, 0.5, 0.6, 0.4, 0.2], info=True),
    )
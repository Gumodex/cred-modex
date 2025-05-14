import itertools

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats

import sklearn.metrics
import sklearn.utils


__all__ = [
    'GoodnessFit'
]


class GoodnessFit:
    @staticmethod
    def ensure_prob_of_class_1(y_pred, prob_base_0=True):
        y_pred = np.array(y_pred)
        return 1 - y_pred if prob_base_0 else y_pred


    @staticmethod
    def r2(y_true:list, y_pred:list, **kwargs):
        value = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred, **kwargs)
        return round(value,4)
    

    @staticmethod
    def r2_adjusted(y_true:list, y_pred:list, n_features:int, **kwargs):
        n = len(y_true)
        if (n <= n_features+1):
            raise ZeroDivisionError(f'"n_features" must be less than {len(y_true)-1}')
        r2 = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred, **kwargs)
        value = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return value
    

    @staticmethod
    def chi2(observed:list, expected:list, alpha:float=0.05, info:bool=False):
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
        y_true = np.array(y_true)
        y_pred = np.clip(np.array(y_pred), 1e-10, 1 - 1e-10)
        
        log_likelihood = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        value = np.sum(log_likelihood)

        if (return_individual == True):
            return log_likelihood

        return float(round(value,4))


    @staticmethod
    def deviance(y_true:list, y_pred:list, return_individual:bool=False, prob_base_0:bool=True):
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)

        log_likelihoods = GoodnessFit.log_likelihood(y_true, y_pred, return_individual=True)
        deviance_residuals = np.sqrt(2 * -log_likelihoods)
        value = np.sum(deviance_residuals ** 2)

        if (return_individual == True):
            return deviance_residuals

        return float(round(value,4))


    @staticmethod
    def aic(y_true:list, y_pred:list, n_features:int, prob_base_0:bool=True):
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)
        log_likelihood = GoodnessFit.log_likelihood(y_true, y_pred)
        value = 2 * n_features - 2 * log_likelihood
        return float(round(value,4))


    @staticmethod
    def aic_small_sample(y_true:list, y_pred:list, n_features:int, sample_size:int, prob_base_0:bool=True):
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)
        aic = GoodnessFit.aic(y_true, y_pred, n_features=n_features)
        if sample_size <= n_features + 1:
            raise ValueError(f'"Sample Size" must be at least {n_features+1}')
        value = aic + (2 * n_features * (n_features + 1)) / (sample_size - n_features - 1)
        return float(round(value,4))


    @staticmethod
    def relative_likelihood(aic_values:list) -> list:
        aic_min = np.min(aic_values)
        value = np.exp((aic_min - aic_values) / 2)
        return value


    @staticmethod
    def weighted_average_estimate(aic_values:list, estimates:list):
        relative_likelihoods = GoodnessFit.relative_likelihood(aic_values=aic_values)
        estimates = np.array(estimates)
        relative_likelihoods = np.array(relative_likelihoods)
        value = np.sum(estimates * relative_likelihoods) / np.sum(relative_likelihoods)
        return float(round(value,4))
    

    @staticmethod
    def weighted_estimates_batch(aic_values:list, predictions_per_model:list[list]) -> list:
        predictions_per_person = list(zip(*predictions_per_model))

        weighted_predictions = [
            GoodnessFit.weighted_average_estimate(aic_values, person_preds)
            for person_preds in predictions_per_person
        ]

        return weighted_predictions
    

    @staticmethod
    def bic(y_true:list, y_pred:list, n_features:int, sample_size:int, prob_base_0:bool=True):
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)
        ll = GoodnessFit.log_likelihood(y_true, y_pred)
        value = np.log(sample_size) * n_features - 2 * ll
        return float(round(value, 4))


    @staticmethod
    def likelihood_ratio_test(loglike_simple:float, loglike_complex:float, added_params:int, alpha:float=0.05, info:bool=False):
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
    def wald_test(beta:float, std_error:float, degrees_freedom:int=1, alpha:float=0.05, null_value:float=0, info:bool=False):
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
            print(f'(wss / (n - g)) == 0 | (wss = {wss}) (n = {n}) (g = {g}) | Optimum Might Have Been Achieved')
            return np.inf

        ch = (bss / (g - 1)) / (wss / (n - g))
        
        return float(round(ch,4))


    @staticmethod
    def gini_variance(y_true:list, y_pred:list, info:bool=False, prob_base_0:bool=True, **kwargs):
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
    def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95, prob_base_0:bool=True):
        aucs = []
        y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
        y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        y_pred = GoodnessFit.ensure_prob_of_class_1(y_pred, prob_base_0)

        for _ in range(n_bootstraps):
            indices = sklearn.utils.resample(np.arange(len(y_true)), replace=True)

            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            if len(np.unique(y_true_boot)) < 2:
                continue

            auc = sklearn.metrics.roc_auc_score(y_true_boot, y_pred_boot)
            aucs.append(auc)

        lower = np.percentile(aucs, (1 - alpha) / 2 * 100)
        upper = np.percentile(aucs, (1 + alpha) / 2 * 100)
        return np.mean(aucs), lower, upper






if __name__ == '__main__':
    df = {
        'Grade': [0]*(95+309) + [1]*(187+224) + [2]*(549+299) + [3]*(1409+495) + [4]*(3743+690) + [5]*(4390+424) + [6]*(2008+94) + [7]*(593+8),
        'Y': [0]*95+[1]*309 + [0]*187+[1]*224 + [0]*549+[1]*299 + [0]*1409+[1]*495 + [0]*3743+[1]*690 + [0]*4390+[1]*424 + [0]*2008+[1]*94 + [0]*593+[1]*8,
        'mu': [309/(95+309)]*(95+309) + [224/(187+224)]*(187+224) + [299/(549+299)]*(549+299) + [495/(1409+495)]*(1409+495) + [690/(3743+690)]*(3743+690) + [424/(4390+424)]*(4390+424) + [94/(2008+94)]*(2008+94) + [8/(593+8)]*(593+8)
    }
    df = pd.DataFrame(df)
    print(

        GoodnessFit.likelihood_ratio_test(-50, -47, 2, info=True),
        GoodnessFit.wald_test(-50, -47, 2, info=True),
        # GoodnessFit.likelihood_ratio_test(df[df['Grade'] == 0]['Y'], np.random.random(95+309), n_features=2, sample_size=95),
    )
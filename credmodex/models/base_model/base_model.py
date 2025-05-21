import pandas as pd
import numpy as np
import inspect
from typing import Union
import warnings
import os
from pprint import pprint, pformat
from tabulate import tabulate

import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency

from credmodex.rating import Rating
from credmodex.discriminancy import *
from credmodex.utils import *


class BaseModel:
    """
    Base class for all models with advanced splitting functionality.
    """

    def __init__(self, model:type=None, treatment:type=None, df:pd.DataFrame=None, seed:int=42, doc:str=None,
                 features=None, target=None, predict_type:str=None, time_col:str=None, name:str=None, n_features:int=None):
        if (df is None):
            raise ValueError("DataFrame cannot be None. Input a DataFrame.")
        if (model is None):
            model = LogisticRegression(max_iter=10000, solver='saga')
        if (treatment is None):
            treatment = lambda df: df
        if isinstance(features,str):
            self.features = [features]
        else:
            self.features = features
        if (n_features is None):
            self.n_features = len(self.features)
        else:
            self.n_features = n_features

        self.seed = seed
        np.random.seed(self.seed)

        self.model = model
        self.treatment = treatment
        self.df = df.copy(deep=True)
        self.doc = doc
        self.target = target
        self.time_col = time_col
        self.name = name
        self.predict_type = predict_type

        self.ratings = {}

        if callable(self.model):
            self.model_code = inspect.getsource(self.model)
        else:
            self.model_code = None

        if callable(self.treatment):
            self.treatment_code = inspect.getsource(self.treatment)
        else:
            self.treatment_code = None

        if callable(self.doc):
            self.doc = inspect.getsource(self.doc)
        else:
            self.doc = None

        self.train_test_()
        self.fit_predict()


    def train_test_(self):
        try:
            self.train = self.df[self.df['split'] == 'train']
            self.test = self.df[self.df['split'] == 'test']

            transformed_features = [col for col in self.df.columns if col not in ['split', self.target, self.time_col]]
            self.features = transformed_features
            self.n_features = len(transformed_features)

            self.X_train = self.df[self.df['split'] == 'train'][self.features]
            self.X_test = self.df[self.df['split'] == 'test'][self.features]
            self.y_train = self.df[self.df['split'] == 'train'][self.target]
            self.y_test = self.df[self.df['split'] == 'test'][self.target]
        except:
            if not getattr(self, 'suppress_warnings', False):
                warnings.warn(
                    'No column ["split"] was found, therefore, the whole `df` will be used in training and testing',
                    category=UserWarning
                )
            self.train = self.df
            self.test = self.df

            transformed_features = [col for col in self.df.columns if col not in ['split', self.target, self.time_col]]
            self.features = transformed_features
            self.n_features = len(transformed_features)

            self.X_train = self.df[self.features]
            self.X_test = self.df[self.features]
            self.y_train = self.df[self.target]
            self.y_test = self.df[self.target]


    def fit_predict(self):
        self.df = self.treatment(self.df)
        self.train_test_()
        predict_type = self.predict_type.lower().strip() if isinstance(self.predict_type, str) else None

        if ('func' in predict_type) or callable(self.model):
            self.df = self.model(self.df)
            return

        self.model = self.model.fit(self.X_train, self.y_train)

        if predict_type and ('prob' in predict_type):
            if hasattr(self.model, 'predict_proba'):
                self.df['score'] = self.model.predict_proba(self.df[self.features])[:,0]
                self.df['score'] = self.df['score'].apply(lambda x: round(x,6))
                return 
            elif hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(self.df[self.features])
                self.df['score'] = scores
                self.df['score'] = self.df['score'].round(6)
                return
            else:
                raise AttributeError("Model doesn't support probability prediction.")
        
        if (predict_type is None) or ('raw' in predict_type):
            if hasattr(self.model, 'predict'):
                preds = self.model.predict(self.df[self.features])
                self.df['score'] = preds
                self.df['score'] = self.df['score'].round(6)
                return
            else:
                raise AttributeError("Model doesn't support raw predictions.")

        else:
            raise SystemError('No ``predict_type`` available')
    

    def add_rating(self, model:type=None, doc:str=None, type='score', optb_type:str='transform', name:str=None,
                   time_col:str=None):
        if (name is None):
            name = f'{model.__class__.__name__}_{len(self.ratings)+1}'
        if (time_col is None):
            time_col = self.time_col
        
        rating = Rating(
            model=model, df=self.df, type=type, features=['score'], target=self.target, 
            optb_type=optb_type, doc=doc, time_col=time_col, name=name
            )
        self.ratings[name] = rating
        setattr(self, name, rating)
        
        # Set the self.rating to the last one defined
        self.rating = rating
        
        return model
    

    def eval_goodness_of_fit(self, method:Union[str,type]='gini', rating:Union[type]=None,
                             comparison_cols:list[str]=[]):
        if rating is None:
            try: rating = self.rating
            except: raise ModuleNotFoundError('There is no model to evaluate!')
        
        if isinstance(method, str): method = method.lower().strip()

        eval_methods = {
            'iv': IV_Discriminant,
            'ks': KS_Discriminant,
            'psi': PSI_Discriminant,
            'gini': GINI_Discriminant,
            'corr': Correlation,
            'good': GoodnessFit,
        }

        # Single method execution
        if method != 'relatory':
            for key, func in eval_methods.items():
                if (key in method) or (method == func):
                    try: return func(df=rating.df, target=self.target, features=['rating'])
                    except: return func
            
        if ('relatory' in method):
            self.rating.plot_gains_per_risk_group().show()
            self.rating.plot_stability_in_time().show()
            try: KS_Discriminant(df=rating.df, target=self.target, features=['rating']).plot().show()
            except: ...
            try: PSI_Discriminant(df=rating.df, target=self.target, features=['rating']).plot().show()
            except: ...
            print('\n=== Kolmogorov Smirnov ===')
            print(KS_Discriminant(df=rating.df, target=self.target, features=['rating']+comparison_cols).table())
            print('\n=== Population Stability ===')
            print(PSI_Discriminant(df=rating.df, target=self.target, features=['rating']+comparison_cols).table())
            print('\n=== Information Value ===')
            print(IV_Discriminant(df=rating.df, target=self.target, features=['rating']+comparison_cols).table())


    def model_relatory_pdf(self, rating:Union[type]=None, add_rating:bool=True,
                           comparison_cols:list[str]=[], pdf:PDF_Report=None, save_pdf:bool=True):
        if (pdf is None):
            pdf = PDF_Report()
        else: ...

        pdf.add_page()
        pdf.main_title(f'Score | {self.name}')

        pdf.chapter_title('Main DataFrame')
        pdf.add_dataframe_split(self.df.head(10), chunk_size=4)

        pdf.add_page()

        try:
            ks = KS_Discriminant(df=self.df, target=self.target, features=['score'] + comparison_cols)

            ks_table = ks.table()
            pdf.chapter_title('Kolmogorov Smirnov')
            pdf.add_dataframe_split(ks_table, chunk_size=4)

            fig = ks.plot()
            fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
            img_path = pdf.save_plotly_to_image(fig)
            pdf.add_image(img_path)
            os.remove(img_path)
        except Exception as e:
            pdf.chapter_df(f"<log> KS failed: {str(e)}")

        try:
            psi = PSI_Discriminant(df=self.df, target=self.target, features=['score'] + comparison_cols)

            psi_table = psi.table()
            pdf.chapter_title('Population Stability Index')
            pdf.add_dataframe_split(psi_table, chunk_size=4)

            fig = psi.plot(add_min_max=[0, 1])
            fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
            img_path = pdf.save_plotly_to_image(fig)
            pdf.add_image(img_path)
            os.remove(img_path)
        except Exception as e:
            pdf.chapter_df(f"<log> PSI failed: {str(e)}")

        try:
            gini = GINI_Discriminant(df=self.df, target=self.target, features=['score'] + comparison_cols)

            pdf.chapter_title('Gini Lorenz Coefficient and Variability')
            gini_var = GoodnessFit.gini_variance(y_pred=self.df['score'], y_true=self.df[self.target], info=True)
            pdf.chapter_df(pformat(gini_var))

            fig = gini.plot()
            fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
            img_path = pdf.save_plotly_to_image(fig)
            pdf.add_image(img_path, w=120)
            os.remove(img_path)
        except Exception as e:
            pdf.chapter_df(f"Plotting failed for {GINI_Discriminant.__name__}: {str(e)}")

        pdf.add_page()

        try:
            iv = IV_Discriminant(df=self.df, target=self.target, features=['score'] + comparison_cols)
            iv_table = tabulate(iv.table().reset_index(drop=False), headers='keys', tablefmt='grid', showindex=False)
            pdf.chapter_title('Information Value')
            pdf.chapter_df(str(iv_table))
        except Exception as e:
            pdf.chapter_df(f"IV Table failed: {str(e)}")

        try:
            hosmer = GoodnessFit.hosmer_lemeshow(y_pred=self.df['score'], y_true=self.df[self.target], info=True)
            pdf.chapter_title('Hosmer Lemeshow')
            pdf.chapter_df(pformat(hosmer))
        except Exception as e:
            pdf.chapter_df(f"Hosmer Lemeshow failed: {str(e)}")

        try:
            deviance = GoodnessFit.deviance_odds(y_pred=self.df['score'], y_true=self.df[self.target], info=True)
            pdf.chapter_title('Deviance Odds')
            pdf.chapter_df(pformat(deviance))
        except Exception as e:
            pdf.chapter_df(f"Deviance Odds failed: {str(e)}")


        if (add_rating == True) and (len(self.ratings.items()) >= 1):
                
            for key, rating in self.ratings.items():
                pdf.reference_name_page = f'{self.name} {rating.name}'
                pdf.add_chapter_rating_page(text1=rating.name, text2=self.name)
                pdf.add_page()
                pdf.main_title(f'Rating | {rating.name}')

                try:
                    pdf.chapter_title('Gains per Risk Group')

                    fig = rating.plot_gains_per_risk_group(width=800)
                    fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
                    img_path = pdf.save_plotly_to_image(fig)
                    pdf.add_image(img_path, w=140)
                    os.remove(img_path)
                except Exception as e:
                    pdf.chapter_df(f"<log> gains per risk failed: {str(e)}")
            
                try:
                    pdf.chapter_title('Stability in Time')

                    fig = rating.plot_stability_in_time(width=800)
                    fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
                    img_path = pdf.save_plotly_to_image(fig)
                    pdf.add_image(img_path, w=140)
                    os.remove(img_path)
                except Exception as e:
                    pdf.chapter_df(f"<log> stability in time failed: {str(e)}")

                pdf.add_page()

                try:
                    ks = KS_Discriminant(df=rating.df, target=rating.target, features=['rating'] + comparison_cols)

                    ks_table = tabulate(ks.table().reset_index(drop=False), headers='keys', tablefmt='grid', showindex=False)
                    pdf.chapter_title('Kolmogorov Smirnov')
                    pdf.chapter_df(str(ks_table))

                    fig = ks.plot(col='rating')
                    fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
                    img_path = pdf.save_plotly_to_image(fig)
                    pdf.add_image(img_path)
                    os.remove(img_path)
                except Exception as e:
                    pdf.chapter_df(f"<log> KS failed: {str(e)}")

                try:
                    psi = PSI_Discriminant(df=rating.df, target=rating.target, features=['rating'] + comparison_cols)

                    psi_table = tabulate(psi.table().reset_index(drop=False), headers='keys', tablefmt='grid', showindex=False)
                    pdf.chapter_title('Population Stability Index')
                    pdf.chapter_df(psi_table)

                    fig = psi.plot(col='rating', discrete=True, sort_index=True)
                    fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
                    img_path = pdf.save_plotly_to_image(fig)
                    pdf.add_image(img_path)
                    os.remove(img_path)
                except Exception as e:
                    pdf.chapter_df(f"<log> PSI failed: {str(e)}")
        
                try:
                    pdf.add_page()
                    iv = IV_Discriminant(df=rating.df, target=rating.target, features=['rating'] + comparison_cols)
                    iv_table = tabulate(iv.table().reset_index(drop=False), headers='keys', tablefmt='grid', showindex=False)
                    pdf.chapter_title('Information Value')
                    pdf.chapter_df(str(iv_table))
                except Exception as e:
                    pdf.chapter_df(f"IV Table failed: {str(e)}")

        if (save_pdf == True):
            pdf.output(f"{self.name}_report.pdf")
        else:
            return pdf
        

    def eval_best_rating(self, sort:str=None):
        
        metrics_dict = {}

        for rating_name, rating in self.ratings.items():
            y_true = rating.df[rating.target]
            y_pred = rating.df['rating']

            iv = IV_Discriminant(rating.df, rating.target, ['rating']).value('rating', final_value=True)
            ks = KS_Discriminant(rating.df, rating.target, ['rating']).value('rating', final_value=True)
            psi = PSI_Discriminant(rating.df, rating.target, ['rating']).value('rating', final_value=True)
            gini = GINI_Discriminant(rating.df, rating.target, ['rating']).value('rating', final_value=True)/100
            auc = round((gini+1)/2, 4)

            contingency_table = pd.crosstab(y_pred, y_true)
            chi2_stat, p_val_chi2, _, _ = chi2_contingency(contingency_table)
            if (p_val_chi2 < 0.05): chi2 = 'Significant Discr.'
            else: chi2 = 'No Significant Discr.'

            y_true = rating.df[rating.target]
            y_pred = rating.df.groupby('rating')[self.target].transform('mean')

            hosmer_lemershow = GoodnessFit.hosmer_lemeshow(y_true=y_true, y_pred=y_pred, info=True)['conclusion']
            log_likelihood = GoodnessFit.log_likelihood(y_true=y_true, y_pred=y_pred)
            aic = GoodnessFit.aic(y_true=y_true, y_pred=y_pred, n_features=self.n_features)
            bic = GoodnessFit.bic(y_true=y_true, y_pred=y_pred, n_features=self.n_features, sample_size=len(self.df))
            wald_test = GoodnessFit.wald_test(y_true=y_true, y_pred=y_pred, info=True)['conclusion']
            deviance_odds = GoodnessFit.deviance_odds(y_true=y_true, y_pred=y_pred, info=True)['power']

            metrics_dict[f'{self.name}.{rating_name}'] = {
                'iv': round(iv,4),
                'ks': round(ks,4),
                'psi': round(psi,4),
                'auc': round(auc,4),
                'gini': round(gini,4),
                'chi2': chi2,
                'wald test': wald_test,
                'log-likelihood': round(log_likelihood,1),
                'aic': round(aic,1),
                'bic': round(bic,1),
            }

        # Create DataFrame and transpose it so rating names are columns
        dff = pd.DataFrame(metrics_dict)
        dff.loc['relative likelihood',:] = GoodnessFit.relative_likelihood(aic_values=list(dff.loc['aic',:].values))
        dff = dff.loc[['relative likelihood'] + [i for i in dff.index if i != 'relative likelihood']]

        try:
            sort = sort.lower()
            if (sort is not None) and (sort in [s.lower() for s in dff.index]):
                dff = dff.T.sort_values(by=sort, ascending=False).T
        except:
            ...

        return dff

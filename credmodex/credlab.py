import sys
import os
import warnings
from typing import Union

from pprint import pprint, pformat
from tabulate import tabulate
import pandas as pd
import numpy as np
import sklearn

sys.path.append(os.path.abspath('.'))
from credmodex.discriminancy import *
from credmodex.models import BaseModel
from credmodex.utils import *

df = pd.read_csv(r'C:\Users\gustavo.filho\Documents\Python\Modules\Credit Risk\test\df.csv')





class CredLab:
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:Union[list[str],str]=None, time_column:str=None,
                 test_size:float=0.1, split_type:str='random', seed:int=42):

        if isinstance(features,str):
            features = [features]

        if (df is None):
            raise ValueError("DataFrame cannot be None")
        self.raw_df = df
        if (not isinstance(target, str)) or (not isinstance(features, list)):
            raise ValueError("target must be a string and features must be a list of strings")
        # The self.df contains only the columns target + features
        features = [f for f in features if f in df.columns and f != target and f != time_column]

        if time_column:
            self.df = df[features + [target] + [time_column]] if features and target else None
        else:
            self.df = df[features + [target]] if features and target else None
        if (self.df is None):
            raise ValueError("Both target and [features] must be provided.")
        
        self.target = target
        self.features = features
        self.seed = seed
        np.random.seed(self.seed)

        self.models = {}
        self.predictions = {}
        self.metrics = {}

        self.test_size = test_size
        self.time_column = time_column
        if self.time_column is None:
            split_type = 'random'
        self.split_type = split_type
        self.train_test_split()

    
    def train_test_split(self):
        self.df = self.df.copy()
        # If random split is selected
        if self.split_type == 'random':
            X = self.df.index.to_list()
            train, test = sklearn.model_selection.train_test_split(X, test_size=self.test_size, random_state=self.seed)
        # If time-based split is selected
        elif self.split_type == 'time':
            if self.time_column is None:
                raise ValueError("A time column must be specified for time-based splitting.")
            
            # Ensure the time column is in datetime format
            self.df[self.time_column] = pd.to_datetime(self.df[self.time_column])

            # Sort by time column to ensure proper order
            self.df = self.df.sort_values(by=[self.time_column])

            # Define the cut-off date for the split
            self.df['scaled'] = (self.df[self.time_column] - self.df[self.time_column].min()) / (self.df[self.time_column].max() - self.df[self.time_column].min())
            target_data_value = self.df[self.time_column].min() + (1-self.test_size) * (self.df[self.time_column].max() - self.df[self.time_column].min())
            cutoff_date = (self.df[self.time_column] - target_data_value).abs().idxmin()
            cutoff_date = self.df.loc[cutoff_date, self.time_column]

            # Filter data for the training and testing sets based on the cutoff date
            train = self.df[self.df[self.time_column] <= cutoff_date]
            test = self.df[self.df[self.time_column] > cutoff_date]

            # Separate features and target
            train = train.index.to_list()
            test = test.index.to_list()

        self.df.loc[train, 'split'] = 'train'
        self.df.loc[test, 'split'] = 'test'

        self.train = self.df[self.df['split'] == 'train']
        self.test = self.df[self.df['split'] == 'test']

        self.X_train = self.df[self.df['split'] == 'train'][self.features]
        self.X_test = self.df[self.df['split'] == 'test'][self.features]
        self.y_train = self.df[self.df['split'] == 'train'][self.target]
        self.y_test = self.df[self.df['split'] == 'test'][self.target]
        

    def plot_train_test_split(self, graph_lib:str='plotly', freq='%Y-%m', width:int=900, height:int=450):
        self.df.loc[:, 'id'] = 1
        self.grouped = self.df.groupby([pd.to_datetime(self.df[self.time_column]).dt.strftime(f'{freq}'),'split']).agg({'id': 'count'}).reset_index()
        train = self.grouped[self.grouped['split'] == 'train']
        test = self.grouped[self.grouped['split'] == 'test']

        if graph_lib == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=train[self.time_column],
                y=train['id'],
                name='Train', marker=dict(color='#292929')
            ))
            fig.add_trace(go.Bar(
                x=test[self.time_column],
                y=test['id'],
                name='Test', marker=dict(color='#704cae')
            ))
            from credmodex.utils.design import plotly_main_layout
            plotly_main_layout(fig, title='Train-Test Split', x='Time', y='Count', 
                height=height, width=width, barmode='stack'
            )
            return fig
        elif graph_lib == 'matplotlib':
            raise NotImplementedError("Matplotlib plotting is not implemented yet.")


    def add_model(self, model:type=None, treatment:type=None, name:str=None, doc:str=None, time_col:str=None, seed:int=42):

        if df is None:
            raise ValueError("DataFrame cannot be None. Input a DataFrame.")
        if name is None:
            name = f'{model.__class__.__name__}_{len(self.models)+1}'
        if time_col is None:
            try: time_col = self.time_column
            except: ...
        
        base_model = BaseModel(
            model=model, treatment=treatment, df=self.df, doc=doc, seed=seed,
            features=self.features, target=self.target, predict_type='prob', time_col=time_col,
            name=name
            )
        self.models[name] = base_model
        setattr(self, name, base_model)

        #self.model is always the last model added!
        self.model = base_model
        
        return model


    def eval_discriminancy(self, method:Union[str,type]='iv', conditions:list=[]):
        if method is None:
            raise ValueError("Method cannot be None. Input a str or a Discriminancy class.")

        df = self.df.copy()
        for condition in conditions:
            df = df.query(condition)
        if df.empty:
            print("DataFrame is empty after applying conditions!")
            return None
        
        if isinstance(method, str): method = method.lower().strip()

        if ('iv' in method) or (method == IV_Discriminant):
            return IV_Discriminant(df, self.target, self.features)
        
        if ('ks' in method) or (method == KS_Discriminant):
            return KS_Discriminant(df, self.target, self.features)
        
        if ('psi' in method) or (method == PSI_Discriminant):
            return PSI_Discriminant(df, self.target, self.features)
        
        if ('gini' in method) or (method == GINI_LORENZ_Discriminant):
            return GINI_LORENZ_Discriminant(df, self.target, self.features)
        
        if ('chi' in method) or (method == CHI2_Discriminant):
            return CHI2_Discriminant(df, self.target, self.features)
        
        if ('corr' in method) or (method == Correlation):
            return Correlation(df, self.target, self.features)


    def eval_goodness_of_fit(self, method:Union[str,type]='gini', model:Union[type]=None,
                             comparinson_cols:list[str]=[]):
        if model is None:
            try: model = self.model
            except: raise ModuleNotFoundError('There is no model to evaluate!')
        
        if isinstance(method, str): method = method.lower().strip()

        eval_methods = {
            'iv': IV_Discriminant,
            'ks': KS_Discriminant,
            'psi': PSI_Discriminant,
            'gini': GINI_LORENZ_Discriminant,
            'corr': Correlation,
            'good': GoodnessFit,
        }

        # Single method execution
        if method != 'relatory':
            for key, func in eval_methods.items():
                if (key in method) or (method == func):
                    try: return func(df=model.df, target=self.target, features=['score'])
                    except: return func
            
        if ('relatory' in method):
            for func in [KS_Discriminant, PSI_Discriminant, GINI_LORENZ_Discriminant]:
                try: func(df=model.df, target=self.target, features=['score']).plot().show()
                except: ...
            print('\n=== Kolmogorov Smirnov ===')
            print(KS_Discriminant(df=model.df, target=self.target, features=['score']+comparinson_cols).table())
            print('\n=== Population Stability ===')
            print(PSI_Discriminant(df=model.df, target=self.target, features=['score']+comparinson_cols).table())
            print('\n=== Gini Lorenz ===')
            print(GINI_LORENZ_Discriminant(df=model.df, target=self.target, features=['score']+comparinson_cols).table())
            print('\n=== Information Value ===')
            print(IV_Discriminant(df=model.df, target=self.target, features=['score']+comparinson_cols).table())
            print('\n=== Hosmer Lemeshow ===') 
            pprint(GoodnessFit.hosmer_lemeshow(y_pred=model.df['score'], y_true=model.df[model.target], info=True))
            print('\n=== Deviance Odds ===') 
            pprint(GoodnessFit.deviance_odds(y_pred=model.df['score'], y_true=model.df[model.target], info=True))
            print('\n=== Gini Variance ===') 
            pprint(GoodnessFit.gini_variance(y_pred=model.df['score'], y_true=model.df[model.target], info=True))


    def model_relatory(self, model:Union[type]=None, rating:Union[type]=None,
                       comparinson_cols:list[str]=[]):
        if model is None:
            try: model = self.model
            except: raise ModuleNotFoundError('There is no model to evaluate!')
        
        print(f'{'':=^100}\n{' SCORE ':=^100}\n{'':=^100}')
        for func in [KS_Discriminant, PSI_Discriminant, GINI_LORENZ_Discriminant]:
            try: func(df=model.df, target=self.target, features=['score']).plot().show()
            except: ...
        print('\n=== Kolmogorov Smirnov ===')
        print(KS_Discriminant(df=model.df, target=self.target, features=['score']+comparinson_cols).table())
        print('\n=== Population Stability ===')
        print(PSI_Discriminant(df=model.df, target=self.target, features=['score']+comparinson_cols).table())
        print('\n=== Gini Lorenz ===')
        print(GINI_LORENZ_Discriminant(df=model.df, target=self.target, features=['score']+comparinson_cols).table())
        print('\n=== Information Value ===')
        print(IV_Discriminant(df=model.df, target=self.target, features=['score']+comparinson_cols).table())
        print('\n=== Hosmer Lemeshow ===') 
        pprint(GoodnessFit.hosmer_lemeshow(y_pred=model.df['score'], y_true=model.df[model.target], info=True))
        print('\n=== Deviance Odds ===') 
        pprint(GoodnessFit.deviance_odds(y_pred=model.df['score'], y_true=model.df[model.target], info=True))
        print('\n=== Gini Variance ===') 
        pprint(GoodnessFit.gini_variance(y_pred=model.df['score'], y_true=model.df[model.target], info=True))

        if rating is None:
            try: rating = self.model.rating
            except: 
                print('There is no rating to evaluate!')
                return

        for key, rating in self.model.ratings.items():
            print(f'\n{'':=^100}\n{f' {key} ':=^100}\n{'':=^100}')

            rating.plot_gains_per_risk_group().show()
            rating.plot_stability_in_time().show()
            try: KS_Discriminant(df=rating.df, target=self.target, features=['rating']).plot().show()
            except: ...
            try: PSI_Discriminant(df=rating.df, target=self.target, features=['rating']).plot().show()
            except: ...
            print('\n=== Kolmogorov Smirnov ===')
            print(KS_Discriminant(df=rating.df, target=self.target, features=['rating']+comparinson_cols).table())
            print('\n=== Population Stability ===')
            print(PSI_Discriminant(df=rating.df, target=self.target, features=['rating']+comparinson_cols).table())
            print('\n=== Information Value ===')
            print(IV_Discriminant(df=rating.df, target=self.target, features=['rating']+comparinson_cols).table())


    def model_relatory_(self, model:Union[type]=None, rating:Union[type]=None,
                       comparinson_cols:list[str]=[]):
        if model is None:
            try: model = self.model
            except: raise ModuleNotFoundError('There is no model to evaluate!')

        pdf = PDF_Report()
        pdf.add_page()

        pdf.main_title(f'Score | {self.model.name}')

        try:
            ks = KS_Discriminant(df=model.df, target=self.target, features=['score'] + comparinson_cols)

            ks_table = tabulate(ks.table().reset_index(drop=False), headers='keys', tablefmt='grid', showindex=False)
            pdf.chapter_title('Kolmogorov Smirnov')
            pdf.chapter_df(str(ks_table))

            fig = ks.plot()
            fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
            img_path = pdf.save_plotly_to_image(fig)
            pdf.add_image(img_path)
            os.remove(img_path)
        except Exception as e:
            pdf.chapter_df(f"<log> KS failed: {str(e)}")

        try:
            psi = PSI_Discriminant(df=model.df, target=self.target, features=['score'] + comparinson_cols)

            psi_table = tabulate(psi.table().reset_index(drop=False), headers='keys', tablefmt='grid', showindex=False)
            pdf.chapter_title('Population Stability Index')
            pdf.chapter_df(psi_table)

            fig = psi.plot()
            fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
            img_path = pdf.save_plotly_to_image(fig)
            pdf.add_image(img_path)
            os.remove(img_path)
        except Exception as e:
            pdf.chapter_df(f"<log> PSI failed: {str(e)}")

        try:
            gini = GINI_LORENZ_Discriminant(df=model.df, target=self.target, features=['score'] + comparinson_cols)

            pdf.chapter_title('Gini Lorenz Coefficient and Variability')
            gini_var = GoodnessFit.gini_variance(y_pred=model.df['score'], y_true=model.df[model.target], info=True)
            pdf.chapter_df(pformat(gini_var))

            fig = gini.plot()
            fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
            img_path = pdf.save_plotly_to_image(fig)
            pdf.add_image(img_path, w=120)
            os.remove(img_path)
        except Exception as e:
            pdf.chapter_df(f"Plotting failed for {GINI_LORENZ_Discriminant.__name__}: {str(e)}")

        pdf.add_page()

        try:
            iv = IV_Discriminant(df=model.df, target=self.target, features=['score'] + comparinson_cols)
            iv_table = tabulate(iv.table().reset_index(drop=False), headers='keys', tablefmt='grid', showindex=False)
            pdf.chapter_title('Information Value')
            pdf.chapter_df(str(iv_table))
        except Exception as e:
            pdf.chapter_df(f"IV Table failed: {str(e)}")

        try:
            hosmer = GoodnessFit.hosmer_lemeshow(y_pred=model.df['score'], y_true=model.df[model.target], info=True)
            pdf.chapter_title('Hosmer Lemeshow')
            pdf.chapter_df(pformat(hosmer))
        except Exception as e:
            pdf.chapter_df(f"Hosmer Lemeshow failed: {str(e)}")

        try:
            deviance = GoodnessFit.deviance_odds(y_pred=model.df['score'], y_true=model.df[model.target], info=True)
            pdf.chapter_title('Deviance Odds')
            pdf.chapter_df(pformat(deviance))
        except Exception as e:
            pdf.chapter_df(f"Deviance Odds failed: {str(e)}")


        if rating is None:
            try: rating = self.model.rating
            except: 
                print('There is no rating to evaluate!')
                return
            
        for key, rating in self.model.ratings.items():
            pdf.add_page()
            pdf.main_title(f'Rating | {rating.name}')

            try:
                pdf.chapter_title('Gains per Risk Group')

                fig = rating.plot_gains_per_risk_group()
                fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
                img_path = pdf.save_plotly_to_image(fig)
                pdf.add_image(img_path, w=120)
                os.remove(img_path)
            except Exception as e:
                pdf.chapter_df(f"<log> gains per risk failed: {str(e)}")
        
            try:
                pdf.chapter_title('Stability in Time')

                fig = rating.plot_stability_in_time()
                fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
                img_path = pdf.save_plotly_to_image(fig)
                pdf.add_image(img_path, w=120)
                os.remove(img_path)
            except Exception as e:
                pdf.chapter_df(f"<log> stability in time failed: {str(e)}")

            pdf.add_page()

            try:
                ks = KS_Discriminant(df=rating.df, target=rating.target, features=['rating'] + comparinson_cols)

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
                psi = PSI_Discriminant(df=rating.df, target=rating.target, features=['rating'] + comparinson_cols)

                psi_table = tabulate(psi.table().reset_index(drop=False), headers='keys', tablefmt='grid', showindex=False)
                pdf.chapter_title('Population Stability Index')
                pdf.chapter_df(psi_table)

                fig = psi.plot(col='rating', discrete=True)
                fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
                img_path = pdf.save_plotly_to_image(fig)
                pdf.add_image(img_path)
                os.remove(img_path)
            except Exception as e:
                pdf.chapter_df(f"<log> PSI failed: {str(e)}")
    
            try:
                iv = IV_Discriminant(df=rating.df, target=rating.target, features=['rating'] + comparinson_cols)
                iv_table = tabulate(iv.table().reset_index(drop=False), headers='keys', tablefmt='grid', showindex=False)
                pdf.chapter_title('Information Value')
                pdf.chapter_df(str(iv_table))
            except Exception as e:
                pdf.chapter_df(f"IV Table failed: {str(e)}")

        pdf.output(f"{self.model.name}_report.pdf")








if __name__ == "__main__":
    print(f'{'':=^100}\n{' SCORE ':=^100}\n{'':=^100}')
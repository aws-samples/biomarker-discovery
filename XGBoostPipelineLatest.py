import xgboost as xgb
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import pandas as pd
import sklearn
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from sklearn.preprocessing import Normalizer
import numpy as np
from copy import deepcopy
from cox_functions import *

normalizer = Normalizer()

default_param_bounds = {
    "max_depth" : (3,10), # originally 3,10
    "eta" : (.01, 1),
    "gamma" : (0.0, 1),
    "max_delta_step" : (1,25), #originally 1,25
#     "subsample" : (0.0, 1), # new feature
#     "colsample_bytree" : (0.0, 1), # new feature
#     "min_child_weight" : (0.0, 1) # new feature
}

default_boost_rounds = 1000
default_early_stop_rounds = 100
default_json_filepath = "XGBoost_model_test_output.json"

class XGBoostPipeline():
    def __init__(
        self, 
        df, 
        random_state,
        label_column,
        num_classes,
        weighted,
        n_iter,
        model_name,
        test_size=0.2,
        param_bounds=default_param_bounds,
        num_boost_rounds=default_boost_rounds,
        early_stopping_rounds=default_early_stop_rounds,
        json_filepath=default_json_filepath,
        dataset_name=None,
        perform_cox=False,
        duration=None,
        event=None,
        all_genes=[],
        use_significant_genes=False
    ):
        print("Initializing pipeline:")
        self.df = df
        self.random_state = random_state
        self.label_column = label_column
        self.num_classes = num_classes
        self.weighted = weighted
        self.n_iter = n_iter
        self.model_name = model_name
        self.test_size = test_size
        self.param_bounds = param_bounds
        self.num_boost_rounds = num_boost_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.json_filepath = json_filepath
        self.dataset_name = dataset_name
        self.perform_cox = perform_cox
        self.duration = duration
        self.event = event
        self.all_genes = all_genes
        self.use_significant_genes = use_significant_genes
        
        return

    def initialize_train_test_data(self):
#         self.remove_empty_features()

        print("Initializing test and train data:")
        y = self.df.pop(self.label_column)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df, 
            y,
            stratify=y,
            random_state=self.random_state, 
            test_size=self.test_size
        )
        
        self.y_train = self.y_train.astype('category')
        self.y_test = self.y_test.astype('category')
        
        if self.perform_cox:
            print("Performing Cox PH pipeline functions:")
#             print("self.all_genes", self.all_genes)
            self.X_train = normalize_gene_expression(self.X_train, self.all_genes)
            self.X_test = normalize_gene_expression(self.X_test, self.all_genes)

            self.X_train = categorize_expression_levels(self.X_train, self.all_genes)
            self.X_test = categorize_expression_levels(self.X_test, self.all_genes)
            
            # This helps us find which patients are normal patients (mapped to 0
            # before feeding into this pipeline) before feeding the data into 
            # Cox PH test.  Since there is no data on survival for normal patients,
            # Cox's PH will break because of missing data.  We might be able to assume that
            # normal patients did not die, but we still wouldn't have  data for the "duration"
            # column which is also necessary for Cox's PH via the open-source lifelines package
            normal_filter_train = self.y_train == 0 
            
            self.info_map, self.significant_genes = cox_ph_pipeline(
                self.X_train[~normal_filter_train], 
                self.all_genes, 
                dataset_name=self.dataset_name, 
                duration=self.duration, 
                event=self.event
            )
            
            if self.use_significant_genes:
                self.X_train = self.X_train[self.significant_genes]
                self.X_test = self.X_test[self.significant_genes]
            

        if self.weighted:
            self.train_weights = class_weight.compute_sample_weight(
                class_weight='balanced',
                y=self.y_train
            )

            self.test_weights = class_weight.compute_sample_weight(
                class_weight='balanced',
                y=self.y_test
            )

            self.dtrain = xgb.DMatrix(
                self.X_train, 
                label=self.y_train, 
                weight=self.train_weights,
                enable_categorical=True,
            )
            
            self.dtest = xgb.DMatrix(
                self.X_test, 
                label=self.y_test, 
                weight=self.test_weights,
                enable_categorical=True,
            )

        else:
            self.dtrain = xgb.DMatrix(
                self.X_train, 
                label=self.y_train,
                enable_categorical=True,
            )
            
            self.dtest = xgb.DMatrix(
                self.X_test, 
                label=self.y_test,
                enable_categorical=True,
            )
        

    def remove_empty_features(self):
        '''Remove empty feature columns from omics dataframes'''
        
        empty_features = []
        
        for col in self.df.columns:
            if (self.df[col].sum() == 0):
                empty_features.append(col)

        percent_empty = (len(empty_features) / len(self.df.columns)) * 100
        print("Removing {} features that have no expression readings. ({}%)".format(len(empty_features),percent_empty))
        
        self.df.drop(
            columns=empty_features, 
            inplace=True
        )

        return

    def xgboost_hyper_param(
        self,
        max_depth,
        eta,
        gamma,
        max_delta_step,
        num_class=3,
        cv=10,
        scoring='roc_auc_ovo',
        random_state=60,
        subsample=1,
        colsample_bytree=1,
        min_child_weight=1,
    ):

#         max_depth = round(max_depth)
        max_depth = int(max_depth)

        self.clf = XGBClassifier(
            use_label_encoder=False,
            objective="multi:softprob",
            eval_metric="auc",
            num_class=self.num_classes,
            min_child_weight=min_child_weight,
            max_depth=max_depth,
            eta=eta,
            max_delta_step=max_delta_step,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=self.random_state
        )


        return np.mean(
            cross_val_score(
                self.clf, 
                self.X_train, 
                self.y_train, 
                cv=cv, 
                scoring=scoring
            )
        )

    def bayesian_optimization(
        self, 
        function, 
        param_bounds, 
        random_state, 
        n_iter
    ):
        
        optimizer = BayesianOptimization(
            f=self.xgboost_hyper_param,
            pbounds=param_bounds, 
            verbose=10,
            random_state=random_state
        )

        optimizer.maximize(
            init_points=3, 
            n_iter=n_iter, 
            acq='ei', 
            kappa=3
        )
        
        return optimizer

    def run_workflow(self):
        '''This function will perform a full XGBoost workflow including:
           1. Splitting data into train and test sets
           2. Creating class weights (if requested)
           3. Categorize genes by expression levels and perform Cox's PH test on each gene (for the train set) if requested.  
               This will also create a json output file and CSV to easily find significant genes and their models
           4. Create an XGBoost model and hyperparameter tune it with Bayesian Optimization
           5. Create a json output file of model metadata
        ''' 
                
        self.initialize_train_test_data()
        
        print("Running XGBoost pipeline")
        print("Beginning Bayesian Optimization:\n")
        self.optimizer = self.bayesian_optimization(
            self.xgboost_hyper_param, 
            self.param_bounds, 
            self.random_state, 
            self.n_iter
        )

        self.best_params = self.optimizer.max['params']
        self.best_auc = self.optimizer.max['target']

        print("Best AUC:", self.best_auc)
        print("Best parameters:", self.best_params)

#         self.best_params['max_depth'] = round(self.best_params['max_depth'])
        self.best_params['max_depth'] = int(self.best_params['max_depth'])
        self.best_params['eval_metric'] = "auc"
        self.best_params['objective'] = "multi:softprob"
        self.best_params['num_class'] = self.num_classes

        self.best_params['min_child_weight'] = 1
        self.best_params['subsample'] = 1
        self.best_params['colsample_bytree'] = 1

        print(self.best_params)

        print("Training XGBoost model")
        
        self.best_model = xgb.train(
            self.best_params,
            self.dtrain,
            num_boost_round=self.num_boost_rounds,
            evals=[(self.dtest, "Test")],
            early_stopping_rounds=self.early_stopping_rounds
        )

        self.y_pred = self.best_model.predict(self.dtest)
        self.predictions = np.argmax(self.y_pred, axis=1)

        self.roc_auc_ovr = roc_auc_score(
            self.y_test.values,
            self.y_pred, 
            multi_class='ovr'
        )
        
        self.roc_auc_ovo = roc_auc_score(
            self.y_test.values,
            self.y_pred, 
            multi_class='ovo'
        )
        
        self.accuracy = accuracy_score(
            self.y_test,
            self.predictions
        )
        
        self.model_filepath = "../final_results/XGBoost/" + self.model_name
        print("Saving model to: ", self.model_filepath)

        self.best_model.save_model(self.model_filepath)
        
        self.populate_json_outfile()
        self.save_json()
        self.save_importance_matrix()
              
        self.print_summary()
        
        return self.best_model 
    
    def print_summary(self):

        print('Results summary:')
        print('Parameter bounds:', self.param_bounds)
        print('Number of boosting rounds:', self.num_boost_rounds)
        print('Early stopping rounds:', self.early_stopping_rounds)
        print('Accuracy =', self.accuracy)
        print('ROC AUC OVO =', self.roc_auc_ovo)
        print('ROC AUC OVR =', self.roc_auc_ovr)
        print('Model filepath =', self.model_filepath)
        print('Importance matrix filepath =', self.importance_matrix_filepath)

        return
    
    def populate_json_outfile(self):
        
        self.model_output = {}
        self.model_output['metadata'] = {}
        self.model_output['metadata']['param_bounds'] = self.param_bounds
        self.model_output['metadata']['num_boost_rounds'] = self.num_boost_rounds
        self.model_output['metadata']['early_stopping_rounds'] = self.early_stopping_rounds
        
        self.model_output['evaluation_metrics'] = {}
        self.model_output['evaluation_metrics']['accuracy'] = self.accuracy
        self.model_output['evaluation_metrics']['roc_auc_ovo'] = self.roc_auc_ovo
        self.model_output['evaluation_metrics']['roc_auc_ovr'] = self.roc_auc_ovr
        
        self.model_output['filepath'] = self.model_filepath
        
        return
    
    def save_json(self):
        
        self.json_filepath = "../final_results/XGBoost/" + self.json_filepath
    
        print("Saving XGBoost JSON results to:", self.json_filepath)

        f = open(self.json_filepath, "w")
        json.dump(self.model_output, f)
        f.close()

#         print("JSON results:")
#         print(json.dumps(self.model_output, indent=4))

        return
    
    def save_importance_matrix(self):
        
        print("Creating importance matrix")
        
        feature_importance_dict = {}
        importance_types = ['gain', 'cover', 'weight', 'total_gain', 'total_cover']
        
        for metric in importance_types:
            feature_importance_dict[metric] = self.best_model.get_score(importance_type=metric)
        
        self.importance_matrix = pd.DataFrame(feature_importance_dict)
        self.importance_matrix_filepath = "../final_results/XGBoost/{}_xgboost_feature_importance.csv".format(self.dataset_name)
        
        print("Saving importance matrix to:", self.importance_matrix_filepath)
        self.importance_matrix.to_csv(self.importance_matrix_filepath)
        
        return
import pandas as pd
from collections import defaultdict
import numpy as np
import pickle
from sklearn.linear_model import  LogisticRegression
from sklearn.preprocessing import StandardScaler

def get_genes(cohort:str) :
    
    results_path = f'result/{cohort}_feature.pickle'
    with open(results_path, 'rb') as file:
            results = pickle.load(file)         
    print(results)    
    return results


def parse_reactomeExpression_and_immunotherapyResponse(cohort:str):

    scores = pd.read_csv(f'result/{cohort}_pathNetGene_score.csv', index_col=0)
    scores.reset_index(inplace=True)
    scores.rename(columns={'index': 'genes'}, inplace=True)
    scores = scores.dropna()
  
    response = pd.read_csv(f'data/{cohort}/response.csv', index_col=0)
    response = response[response['response'] != -1]
      
    exp_dic, responses = defaultdict(list), []
    e_samples = []
    for e_sample in scores.columns :
        if e_sample in response['sample'].values :
            responses.append(response[response['sample'] == e_sample]['response'])
            e_samples.append(e_sample)

    responses = np.array(responses)

    scores = scores.loc[:,['genes'] + e_samples]
 
    return e_samples, scores, responses


def ML_hyperparameters():
    model = LogisticRegression()    
    param_grid = {'penalty':['l2'], 'max_iter':[1000], 'solver':['lbfgs'], 'C':np.arange(0.1, 1, 0.1), 'class_weight':['balanced'] }
    
    return model, param_grid


## expression scaler
def expression_StandardScaler(exp_df:pd.DataFrame):
	col1 = exp_df.columns[0]
	tmp = StandardScaler().fit_transform(exp_df.T.values[1:])
	new_tmp = defaultdict(list)
	new_tmp[col1] = exp_df[col1].tolist()
	for s_idx, sample in enumerate(exp_df.columns[1:]):
		new_tmp[sample] = tmp[s_idx]
	output = pd.DataFrame(data=new_tmp, columns=exp_df.columns)
	return output


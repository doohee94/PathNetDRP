import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, f1_score, precision_recall_curve,roc_auc_score,average_precision_score,accuracy_score


exec(open('utilities.py').read())

cohorts = ["Gide"]
loo = LeaveOneOut()

for cohort in cohorts :

    edf = pd.DataFrame()
    samples, edf, responses = parse_reactomeExpression_and_immunotherapyResponse(cohort)

    tmp = StandardScaler().fit_transform(edf.T.values[1:])
    new_tmp = defaultdict(list)
    new_tmp['genes'] = edf['genes'].tolist()
    for s_idx, sample in enumerate(edf.columns[1:]):
        new_tmp[sample] = tmp[s_idx]
    edf = pd.DataFrame(data=new_tmp, columns=edf.columns)

    features= get_genes(cohort)

    tmp_edf = edf.loc[edf['genes'].isin(features),:]
    exp = tmp_edf.T.values[1:]
    sample = tmp_edf.columns[1:]
    features_names = tmp_edf['genes'].tolist()


    obs_responses, pred_responses, pred_probabilities,pred_samples = [], [], [],[]

    for train_idx, test_idx in loo.split(exp):
        X_train, X_test, y_train, y_test = exp[train_idx], exp[test_idx], responses[train_idx], responses[test_idx]
        
        y_test = np.ravel(y_test)
        y_train = np.ravel(y_train)
        
        model, param_grid = ML_hyperparameters()
        
        gcv = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=5).fit(X_train, y_train)
        pred_status = gcv.best_estimator_.predict(X_test)[0]
        
        
        obs_responses.append(y_test[0])
        pred_responses.append(pred_status)
        pred_probabilities.append(gcv.best_estimator_.predict_proba(X_test)[0][1])
        pred_samples.append(samples[test_idx[0]])
        

    accuracy = accuracy_score(obs_responses, pred_responses)
    precision, recall, _ = precision_recall_curve(obs_responses, pred_probabilities, pos_label=1)
    fpr_proba, tpr_proba, _ = roc_curve(obs_responses, pred_probabilities, pos_label=1)
    average_precision  = average_precision_score(obs_responses, pred_probabilities)
    F1 = f1_score(obs_responses, pred_responses, pos_label=1)
    auc = roc_auc_score(obs_responses, pred_responses)
        
    print(f'{cohort} \n Accuracy  : {accuracy} \n F1 : {F1}')
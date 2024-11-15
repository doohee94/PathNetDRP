import numpy as np
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier

exec(open('utilities.py').read())

test_cohorts = ['Auslander']
train_cohort = "Gide"

for test_idx, test_cohort in enumerate(test_cohorts):
    train_samples, train_edf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_cohort)
    test_samples, test_edf,  test_responses = parse_reactomeExpression_and_immunotherapyResponse(test_cohort)

    common_genes = list(set(train_edf['genes'].tolist()) & set(test_edf['genes'].tolist()))
    train_edf = train_edf.loc[train_edf['genes'].isin(common_genes),:].sort_values(by='genes')
    test_edf = test_edf.loc[test_edf['genes'].isin(common_genes),:].sort_values(by='genes')

    train_edf = expression_StandardScaler(train_edf)
    test_edf = expression_StandardScaler(test_edf)

    feature = get_genes(train_cohort)

    train_edf = train_edf.loc[train_edf['genes'].isin(feature),:]
    test_edf = test_edf.loc[test_edf['genes'].isin(feature),:]

    X_train, X_test = train_edf.T.values[1:], test_edf.T.values[1:]
    y_train, y_test = train_responses, test_responses
                
    y_test = np.ravel(y_test)
    y_train = np.ravel(y_train)
    
    model = AdaBoostClassifier()
    param_grid = {'n_estimators': [50,100,200],'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],'algorithm': ['SAMME', 'SAMME.R']}
   
    auc_scores = []
    
    for i in range(10) :
        
        sfkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        gcv = GridSearchCV(model, param_grid=param_grid, cv=sfkf, scoring='roc_auc', n_jobs=10).fit(X_train, y_train)
        pred_status = gcv.best_estimator_.predict(X_test)
        pred_proba = gcv.best_estimator_.predict_proba(X_test)[:,1]

        aucs = roc_auc_score(y_test, pred_proba)
        auc_scores.append(aucs)
        
    print(f'{test_cohort} AUC : {round(np.mean(auc_scores), 3)} ± {"{:.2e}".format(np.std(auc_scores))}')

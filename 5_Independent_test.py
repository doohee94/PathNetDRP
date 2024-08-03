import numpy as np
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

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

    # make predictions
    C_param = np.arange(0.1, 1, 0.1)
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=50)

    test_avg = []
    for valid_index, test_index in kf.split(X_test, y_test) :

        models = []
        best = 0
        best_index = 0
        for j, c in enumerate(C_param) :
            model = LogisticRegression(C=c, penalty = 'l2', max_iter=1000, solver='lbfgs', class_weight='balanced')
            model.fit(X_train, y_train)
            models.append(model)
    
            X_valid, Y_valid = X_test[valid_index], y_test[valid_index]
            test_x, test_y = X_test[test_index], y_test[test_index]
    
            valid_pred_proba = model.predict_proba(X_valid)[:,1]
            valid_auc = roc_auc_score(Y_valid, valid_pred_proba) 

            if valid_auc > best :
                best = valid_auc
                best_index = j

        final_model = models[best_index]
        test_pred_proba = final_model.predict_proba(test_x)[:,1]
        test_auc = roc_auc_score(test_y, test_pred_proba) 
        test_avg.append(test_auc)

    print(f'Final AVG AUC = {np.mean(test_avg)}')
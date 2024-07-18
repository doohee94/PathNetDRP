import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pickle


cohorts = ["Gide"]
for cohort in cohorts :

    response = pd.read_csv(f"data/{cohort}/response.csv", index_col=0).set_index("sample")
    response = response[response['response'] != -1]
    r_samples = response[response['response'] == 1].index.values
    nr_samples = response[response['response'] == 0].index.values
        
    gene_expression = pd.read_csv(f'result/{cohort}_pathNetGene_score.csv', index_col=0)
    r_ge = gene_expression[r_samples]
    nr_ge = gene_expression[nr_samples]
        
    g_genes = gene_expression.index.values
        
    results = []
    p_values = []
    for gene in g_genes :
        
        rg = r_ge.loc[gene].values.reshape(-1)
        nrg = nr_ge.loc[gene].values.reshape(-1)
        t_statistic, p_value = stats.ks_2samp(rg, nrg)
        
        
        p_values.append(p_value)
        results.append(gene)

    _, qvalues, _, _ = multipletests(p_values, method='holm-sidak')

    temp_df = pd.DataFrame({'gene':results, 'pval':qvalues})    
    temp_df.sort_values(by=["pval", "gene"], ascending=[True, True], inplace=True)
    # temp_df.sort_values(by=["pval"], ascending=[True], inplace=True)
    temp_df = temp_df.iloc[:30]
    results = list(temp_df['gene'].values)
    
    print(f'========{cohort}')
    for g in results :
        print(g)
    
    with open(f'result/{cohort}_feature.pickle', 'wb') as file:
        pickle.dump(results, file)  

print('3_select_feature end')
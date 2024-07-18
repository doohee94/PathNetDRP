from collections import defaultdict
import pandas as pd
import pickle


cohorts = ["Gide"]

for cohort in cohorts :
    print(f'================{cohort}============') 
    with open(f'result/{cohort}_score_by_pathway.pickle', 'rb') as file:
        data = pickle.load( file)  
    
    response = pd.read_csv(f'data/{cohort}/response.csv', index_col=0)
    samples = response[response['response']!= -1]['sample'].values
    
    genes = defaultdict(int)
    results = dict()
    for key, value in data.items() :
        test = value.iloc[:,0]
        if len(set(test.values)) == 1 :
            continue
        else :
            
            if len(results.keys()) == 0 :
                results = value.to_dict()
                for _, value in results.items() :
                    for t_keys in value.keys() :
                        genes[t_keys] += 1
                    break
                
            else :
                temp = value.to_dict()
                for _, value in temp.items() :
                    for t_keys in value.keys() :
                        genes[t_keys] += 1
                    break
            
                for result_sample, result_v in results.items() :
                    if result_sample not in temp.keys() :
                        continue
                    temp_v = temp[result_sample]
                    for t_genes,t_v in temp_v.items() :     
                                         
                        if t_genes in result_v.keys() :
                            result_v[t_genes] += t_v
                        else :
                            result_v.update({t_genes:t_v})
                          

    for sample, scores in results.items() :
        
        for t_gene, t_value in scores.items() :
            scores[t_gene] = t_value/genes[t_gene]
            
    result_df = pd.DataFrame().from_dict(results)
    
    result_df.to_csv(f'result/{cohort}_pathNetGene_score.csv')
                            
print('2_calculate_score end')
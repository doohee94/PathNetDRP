from collections import defaultdict
import pandas as pd
import pickle
import networkx as nx
from scipy.stats import hypergeom
import networkx as nx
from networkx.algorithms.link_analysis import pagerank
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests


pathways = pd.read_csv('data/reactome_pathway.csv')
pathway_dict = defaultdict(list)

for pathway in pathways.columns.values :
    pp = pathways[pathway].dropna().values
    pathway_dict[pathway].extend(pp)

cohorts = ["Gide"]
targets = {'Gide':["PDCD1","CTLA4"] }

for cohort in cohorts :
    print(f'================{cohort}============') 
    
    save_path = f'data/{cohort}/'

    gene_expression = pd.read_csv(f'data/{cohort}/gene_expression.csv', index_col=0)
    scaled_data =  MinMaxScaler().fit_transform(gene_expression)
    gene_expression = pd.DataFrame(scaled_data, columns=gene_expression.columns, index = gene_expression.index)

    #pagerank
    ppi_network = pd.read_csv(f'data/ppi_network.csv', index_col=0)
    ppi_network = ppi_network[['src','dest']]
    # ppi_network = ppi_network[['dest','src']]
    net_genes = set(ppi_network.values.reshape(-1))
    cohort_genes = list(set(gene_expression.index.values) & set(net_genes))
   
    ppi_network = ppi_network.query('src in @cohort_genes and dest in @cohort_genes')

    net_genes = set(ppi_network.values.reshape(-1))

    propagate_input = {}
    
    for gg in net_genes :
        propagate_input[gg] = 0

        if gg in targets[cohort]  :
            propagate_input[gg] = 1

    # PPI network Propagation 
    G = nx.DiGraph()
    G.add_edges_from(ppi_network[['src','dest']].values)
    # G.add_edges_from(ppi_network[['dest','src']].values)
    propagate_scores = pagerank(G, personalization=propagate_input, max_iter=100, tol=1e-06) 
    sorted_dict = dict(sorted(propagate_scores.items(), key=lambda item: item[1], reverse=True))
    
    pagerank_genes = list(sorted_dict.keys())[:2000]
    
    M = len(cohort_genes)
    N = len(pagerank_genes)
    
    p_values = []
    pps = []
    
    for key, value in pathway_dict.items() :
        pw_genes = list(set(value) & set(cohort_genes))
        n = len(pw_genes)
        k = len(set(pw_genes) & set(pagerank_genes))
            
        p_value = hypergeom.sf(k-1, M, n, N)
        
        p_values.append(p_value)
        pps.append(key)
       
    _, qvalues, _, _ = multipletests(p_values)
    df = pd.DataFrame({'pathway':pps, 'p_value':p_values,'adjust':qvalues})
    df_sorted = df.sort_values(by='adjust')
    df_sorted = df_sorted[df_sorted['adjust']<0.01]

    
    # pathway subnetwork propagation 
    cohort_result = defaultdict(pd.DataFrame)
    for i, row in df_sorted.iterrows() :

        p_name = row['pathway']
        p_genes = pathway_dict[p_name]
        
        p_net = ppi_network.query("src in @p_genes and dest in @p_genes")
      
        p_t_genes = list(set(cohort_genes) & set(p_genes))
        
        p_gene_expression = gene_expression.loc[p_t_genes]
 
        ge_dict = p_gene_expression.to_dict()

        temp_result = defaultdict(dict)
        for sample, ge_values in ge_dict.items() :
            G = nx.DiGraph()
            G.add_edges_from(p_net[['src','dest']].values)
            # G.add_edges_from(p_net[['dest','src']].values)
            try:
                ## NETWORK PROPAGATION
                propagate_scores = pagerank(G, personalization=ge_values, max_iter=100, tol=1e-06) 
                sorted_dict = dict(sorted(propagate_scores.items(), key=lambda item: item[1], reverse=True))
                temp_result[sample] = sorted_dict
            except Exception as e :
                print(e)
                continue
        
        cohort_result[p_name] = pd.DataFrame().from_dict(temp_result)


    with open(f'result/{cohort}_score_by_pathway.pickle', 'wb') as file:
        pickle.dump(cohort_result, file)  
    
print('1_get_score_by_pathway end')


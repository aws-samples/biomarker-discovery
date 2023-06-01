import pandas as pd
import numpy as np
import json
from lifelines import CoxPHFitter
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()

##Functions to create Cox PH Pipeline

def normalize_gene_expression(df, genes):
    
    print("Log-transforming gene expression data")
    df[genes] = np.log(1e-5+ df[genes].astype('float64'))
    
    print("Normalizing gene expression")
    normalized_data = normalizer.fit_transform(df[genes])
    df[genes] = normalized_data
    
    return df

def categorize_expression_levels(df, genes):
    # These are what the categories map to based on relative expression levels:
    # {"Low Expression (below median)" : 1, "High Expression (above median)" : 2,}

    for gene in genes:
        conditions = [
            df[gene] <= df[gene].median(),
        ]

        results = [1]
        df[gene] = np.select(conditions,results,default=2)
        df[gene] = df[gene].astype("category")
    
    return df

def populate_summary(cph_fitter, cph_map, gene, dataset_name):
    
    cph_map[gene]["summary_file"] = "../final_results/{}_{}_kirc_cox_model_summary.csv".format(gene, dataset_name)
    cph_fitter.summary.to_csv(cph_map[gene]["summary_file"])
    
    return

def populate_concordance(cph_fitter, cph_map, gene):
    
    cph_map[gene]["concordance_index"] = deepcopy(cph_fitter.concordance_index_) 
    
    return

def populate_significance(cph_map, gene, sig_genes_list, pvalue=.05):
    
    results = pd.read_csv(cph_map[gene]['summary_file'])
    cph_map[gene]["p-value"] = results.at[0,'p']
    
    if cph_map[gene]["p-value"] >= pvalue:
        cph_map[gene]["significant"] = False
        
        return

    cph_map[gene]["significant"] = True
    sig_genes_list.append(gene)
    
    return

def plot_sig_genes(cph_fitter, cph_map, dataset, gene):
    
    plot_title = "Kidney Cancer Survival Risk: {}".format(gene)
    cph_fitter.plot_partial_effects_on_outcome(
        covariates=gene,
        values=[1,2],
        cmap="coolwarm",
        title=plot_title
    )
    filepath = "../figures/kirc_survival_risk_{}_{}.jpg".format(dataset, gene)
    plt.savefig(filepath,bbox_inches='tight', dpi=150)
    cph_map[gene]["kaplan_meier_plot_filepath"] = filepath
        
    return

def save_json(cph_map, dataset="mRNA"):
    
    filepath = "../final_results/{}_individual_cox_results.json".format(dataset)
    print("Saving Cox PH JSON results to:", filepath)
    
    f = open(filepath, "w")
    json.dump(cph_map, f)
    f.close()
    
#     print("JSON results:")
#     print(json.dumps(cph_map, indent=4))
    
    return

def save_csv(cph_map, dataset="mRNA"):
    
    all_genes_df = pd.DataFrame(cph_map)

    all_genes_df = all_genes_df.T
    sig_genes_filter = all_genes_df["significant"] == True
    sig_genes_df = all_genes_df[sig_genes_filter]
    all_genes_df.to_csv("../final_results/{}_cox_results_table_all_genes.csv".format(dataset))
    sig_genes_df.to_csv("../final_results/{}_cox_results_table_significant_genes.csv".format(dataset))
    
    return

def save_results(cph_map, dataset="mRNA"):
    
    save_json(cph_map, dataset)
    save_csv(cph_map, dataset)
    
    return

def cox_ph_pipeline(df, genes, dataset_name, duration, event, plot_genes=False, pvalue=.05, show_progress=False):
    
    cph = CoxPHFitter(penalizer=0.1)
    cph_map = {} #k: gene_name, v: {info about model}
    sig_genes_list = []
    
    for gene in genes:
#         print("Testing gene:", gene)
        
        subset = [duration, event, gene]
        cph.fit(df[subset], duration_col=duration, event_col=event, show_progress=show_progress, step_size=.01)
        
        cph_map[gene] = {}
        populate_summary(cph, cph_map, gene, dataset_name)
        populate_concordance(cph, cph_map, gene)
        populate_significance(cph_map, gene, sig_genes_list, pvalue)   
        
    if plot_genes:
        for gene in sig_genes_list:
            plot_sig_genes(cph, cph_map, dataset_name, gene)

    save_results(cph_map, dataset_name)
    
    return cph_map, sig_genes_list
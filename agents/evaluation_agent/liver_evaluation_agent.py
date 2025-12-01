import ast
import os
import re
from agents.evaluation_agent.base_evaluation_agent import BaseEvaluationAgent
import anndata as ad
import scanpy as sc
import pandas as pd
from utils.LLM import query_llm
from utils.liver_process_toolkit import find_similar_cluster_pairs, identify_marker_genes, zscore_normalize_expression
from sklearn.preprocessing import MinMaxScaler


class LiverEvaluationAgent(BaseEvaluationAgent):
    def __init__(self, hypothesis, output_dir, input_dir, existing_genes, file_name,model_name = "gpt-4o",model_provider = "openai"):
        super().__init__(hypothesis, output_dir)
        self.input_dir=input_dir
        self.file_name=file_name
        self.adata = ad.read_h5ad(os.path.join(input_dir, file_name))
        self.existing_genes = existing_genes

        self.model_name = model_name
        self.model_provider = model_provider

    def extract_dotplot(self,groupby,dotplot):
        dotplot_data, dotplot_data_frac = pd.DataFrame(), pd.DataFrame()
        dotplot = sc.pl.dotplot(self.adata, self.existing_genes, groupby=groupby, return_fig=True)
        dotplot_data = dotplot.dot_color_df  # Dot colors correspond to expression levels
        dotplot_data_frac = dotplot.dot_size_df  # Dot sizes correspond to the fraction of cells expressing the gene
        return dotplot_data, dotplot_data_frac
    
    def dotplot_algorithm(self,dotplot_data,dotplot_data_frac):
        normalized_dotplot_data = zscore_normalize_expression(dotplot_data)
        marker_genes = identify_marker_genes(normalized_dotplot_data, dotplot_data_frac)
        empty_keys = [key for key, value in marker_genes.items() if not value]
        marker_genes = {key: value for key, value in marker_genes.items() if value}

        similar_cluster_pairs = find_similar_cluster_pairs(dotplot_data, dotplot_data_frac, exp_diff_thresh=0.5, frac_diff_thresh=0.5, max_diff_genes=2, logfc_thresh=1.0)
        similar_clusters_dict = {}
        for cluster1, cluster2, diff_genes in similar_cluster_pairs:
            if diff_genes:  
                key = f"{cluster1} vs {cluster2}"
                similar_clusters_dict[key] = diff_genes

        return marker_genes,empty_keys,similar_clusters_dict

    def gene_level_metrics(self,dotplot_data,dotplot_data_frac):
        merged_data = pd.concat([dotplot_data.add_suffix('_exp'), dotplot_data_frac.add_suffix('_frac')], axis=1)
        scaler = MinMaxScaler()
        for gene in self.existing_genes:
            exp_col = gene + '_exp'
            frac_col = gene + '_frac'
            merged_data[[exp_col, frac_col]] = scaler.fit_transform(merged_data[[exp_col, frac_col]])
        for gene in self.existing_genes:
            exp_col = gene + '_exp'
            frac_col = gene + '_frac'
            merged_data[gene + '_score'] = merged_data[[exp_col, frac_col]].sum(axis=1)
        def get_top_clusters(scores, threshold_ratio=0.7):
            top_clusters = []
            max_score = scores.iloc[0]
            if max_score < 1:
                return top_clusters
            for i, score in enumerate(scores):
                if score >= max_score * threshold_ratio:
                    top_clusters.append(i)
                else:
                    break
            return top_clusters
        top_clusters = {}
        fail_list = []
        success_list = []
        for gene in self.existing_genes:
            gene_score_col = gene + '_score'
            sorted_scores = merged_data[gene_score_col].sort_values(ascending=False)
            top_cluster_indices = get_top_clusters(sorted_scores)
            if len(top_cluster_indices) == 0:
                top_clusters[gene] = []
                fail_list.append(gene)
            else:
                success_list.append(gene)
                top_clusters[gene] = sorted_scores.iloc[top_cluster_indices].index.tolist()
        return success_list,fail_list
        
    def evaluate(self, groupby, dotplot=None,duplet_rule="", contamination_rule="",possible_cell_types = "",iteration=1):
        print("DEBUG: inside evaluate")
        dotplot_data, dotplot_data_frac = self.extract_dotplot(groupby,dotplot)
        marker_genes,empty_keys,similar_clusters_dict=self.dotplot_algorithm(dotplot_data,dotplot_data_frac)
        success_list,fail_list = self.gene_level_metrics(dotplot_data,dotplot_data_frac)
        cluster_size = len(dotplot_data_frac)
        content =  f'''
        Context:
        You are working on a single-cell RNA sequencing cell type annotation task. The goal is to identify distinct cell types based on gene expression patterns. You have created a dotplot using a set of marker genes to visualize gene expression across different clusters.

        MUST remember: you should list the cell types here. All cell types you refer to, should be in here. {possible_cell_types}

        Data:
        For each cluster, the top genes are in this dictionary: {marker_genes}
        The clusters cannot find top genes are: {empty_keys}
        These are the genes that are successfully (highly) expressed in some clusters: {success_list}
        These are the genes that are failed to express in any clusters: {fail_list}
        There are some clusters that have similar expression, so we did differential expression analysis for these cluster pairs. The pairs and top differential genes are in: {similar_clusters_dict}

        Add-on:
        Duplet and Contamination in dotplot: {duplet_rule}
        Any other important instructions: {contamination_rule}

        Please give greatest attention to the Add-on part, if they are provided. Make sure to use them in your analysis.

        Instructions:
        Please analyze the provided data and answer the following questions:

        1. Gene-level analysis:
        a) Which genes are highly expressed in specific clusters? Provide a detailed description.
        b) Are there any genes that show differential expression across clusters (high in some, low in others)?
        c) Are there any genes that are not informative for cell type annotation (low expression across all clusters)?
                You should answer this question based on negation of 1b.

        2. Cluster-level analysis:
        a) Are there any clusters that lack high expression of any marker gene? If so, list the cluster numbers. 
        There are in total {cluster_size} clusters, index from 0.

        3. Overall assessment:
        a) Based on the gene expression patterns, are there distinct clusters that potentially represent different cell types?
        b) Can you assign specific cell type identities to any of the clusters based on the marker gene expression? If so, provide your cell type annotations. You should only assign one cell type to each cluster. No doublet or contamination here.
        c) To refine the cell type annotation, recommend possible additional cell types.
        d) To refine the cell type annotation, recommend any particular cluster to perform subgrouping.

        4. Confidence assessment:
            a) What are confidence levels of your annotation? Please also assign a confidence score to the process name you selected.
            This score should follow the name in parentheses and range from 0.00 to 1.00. A score of 0.00 indicates the
            lowest confidence, while 1.00 reflects the highest confidence. This score helps gauge how accurately the annotation is. 
            Your choices of confidence score should be a normal distribution (average = 0.5)
            You should consider if the annotation is widely seen, using deterministic wording and following background of dataset.
            For instance, if you label a doublet, or using word "probable", the score should be lower.

            b) Based on confidence levels, what are the annotation results that you want to "stabilize", that is not change in next steps? 
            if {iteration} is 1, you should choose top 1/3 confident clusters. if {iteration} is 2 and beyond, you should choose top 2/3 or more clusters. 
            You can choose this threshold.

        Please provide your answers in a structured JSON format, addressing each question separately using "1a" "2a" etc.

        Remember, this is one iteration cell type annotation for a liver scRNA-seq dataset. Your insights will guide further refinement of the analysis.
        '''
        system_role = "You are an expert bioinformatician specializing in single-cell RNA sequencing data analysis and cell type annotation."

        LLM_output = query_llm(content=content,system_role=system_role,model_provider=self.model_provider,model_name=self.model_name)
        evaluation = LLM_output
        self.evaluation_results = evaluation
        return self.evaluation_results, fail_list, success_list, marker_genes, empty_keys, similar_clusters_dict
    
    
    def prediction(self,evaluation):
        content = f'''
        Provided Text: {evaluation}
        Please look at the provided text and summarize the result. You should list the cell type annotation.
        The provided text includes rich data, and here some ways of analysis are provided.
        For example, you will see something like "Alb": ["3", "10", "16", "21", "27"], which means Alb gene is highly expressed in these clusters. 
        As Alb is strongly related to hepatocytes, we can label these clusters as hepatocytes.

        No doublet or contamination here. So we must assign one cell type for each cluster.

        Your output must be a string like this: "0, 'B Cell', 1, 'T Cell', etc", with no more wrapping, where 0 and 1 are cluster number as integers,
        and 'B cell' or 'T cell' are example cell types as strings.
        You should try to list out as many convinced pairs as possible. However, do not include cell type names like "Unknown" or "Unannotated".
        '''
        system_role = "You are an expert in scRNA sequencing cell type annotation."
        LLM_output = query_llm(content=content,system_role=system_role,model_provider=self.model_provider,model_name=self.model_name)
        summary = LLM_output
        return summary
    
    def execution(self,summary):
        content = summary#["choices"][0]["message"]["content"]
        items = content.split(', ')
        '''
        cell_dict = {}
        for i in range(0, len(items), 2):
            try:
                key = int(items[i].strip("```plaintext\n"))
                value = items[i+1].strip("'")
                cell_dict[key] = value
            except (ValueError, IndexError):
                print("Error in parsing cell type prediction, please run again")
                break
        return cell_dict
        '''
        cell_dict = {}
        for i in range(0, len(items), 2):
            try:
                key_raw = items[i]
                value_raw = items[i + 1]
                
                # Use regex to extract numeric key and cleaned label
                key_match = re.search(r'\d+', key_raw)
                value_match = re.search(r"'([^']+)'", value_raw) or re.search(r'"([^"]+)"', value_raw)
                
                if key_match and value_match:
                    key = int(key_match.group())
                    value = value_match.group(1)
                    cell_dict[key] = value
                else:
                    raise ValueError("Invalid formatting")
            except (ValueError, IndexError) as e:
                print(f"Error parsing index {i}: {e}")
                continue
        return cell_dict

    def find_no_gene_cluster(self, evaluation):
        content = evaluation
        start_index = content.find('"2a"')
        value_end_index = content.find('"3a"')
        output_2a = content[start_index:value_end_index]
        numbers = re.findall(r'\d+', output_2a)
        no_gene_clusters = [int(num) for num in numbers]

        return no_gene_clusters
#!/usr/bin/env python
import ast
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import scanpy as sc
from typing import Dict, List

from utils.LLM import query_llm 

class CellAnnotationPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.initialize_directories()
        self.load_data()
        
        # State variables
        self.current_iteration: int = 0
        self.annotation_dict: Dict = {}
        self.no_gene_cluster: List = []
        self.evaluation: str = ""
        self.failed_genes: List = []
        self.successful_genes: List = []
        
    def initialize_directories(self):
        os.makedirs(self.config['input_dir'], exist_ok=True)
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
    def load_data(self):
        """Load input data and prepare markers"""
        h5ad_path = os.path.join(self.config['input_dir'], self.config['h5ad_file'])
        self.adata = sc.read_h5ad(h5ad_path)
        self.adata.obs[self.config['original_grouping']] = self.adata.obs[self.config['original_grouping']].astype('category')
        
    def run_iteration(self):
        """Run one complete iteration of the pipeline"""

        df = pd.read_csv(os.path.join(self.config['input_dir'], self.config['markers_file']))
        cluster_gene_dict = df.groupby('cluster')['gene'].apply(list).to_dict()

        system_role = "You are expert in scRNA sequencing cell type annotation."
        content = f'''
                this is background information: {self.config["initial_hypothesis"]}
                look at this dict: {cluster_gene_dict}. This is cluster number and the corresponding top differential genes of each cluster. Please provide cell type annotation for each cluster. 
                Output in text dict format just like the input dict. Keys are number of cluster, and Values are strings of cell type names. Output should be text dict, no other word should exist.
        '''
        reply = query_llm(content=content,system_role=system_role,model_name=self.config["model_name"],model_provider=self.config["model_provider"])
        sanitized_str = reply.replace("```", "")
        annotation_dict = None
        try:
            annotation_dict = ast.literal_eval(sanitized_str)
        except:
            pass

        annotation_dict = {0: 'T cells', 1: 'B cells', 2: 'Monocytes', 3: 'CD8 T cells', 4: 'CD8 T cells', 5: 'Monocytes', 6: 'Dendritic cells', 7: 'Platelets'}
        print("[DEBUG] auto fill in result\n",annotation_dict)
        self.annotation_dict = annotation_dict
        
        output_column_name=self.config["output_column"]

        annotation_dict = self.annotation_dict
        groupby = self.config['original_grouping']
        org_dict = {int(i): str(i) for i in self.adata.obs[groupby]}
        org_dict.update(annotation_dict)
        #self.adata.obs[groupby] = self.adata.obs[groupby].astype(int)
        self.adata.obs[output_column_name] = self.adata.obs[groupby].map(org_dict).astype('category')
        self.adata.obs[groupby] = self.adata.obs[groupby].astype("category")

        self.adata.write(os.path.join(self.config['input_dir'], self.config['h5ad_file']))
        pd.Series(self.annotation_dict).to_csv(
            os.path.join(self.config['output_dir'], f'basic_annotations.csv')
        )

    def run_pipeline(self, iterations: int = 3):
        """Run complete pipeline with specified number of iterations"""
        for _ in range(iterations):
            print(f"Current iteration {self.current_iteration}")
            self.run_iteration()
            print(f"Current annotations: {self.annotation_dict}")


if __name__ == "__main__":
    config = {
        "model_name":"gpt-4o-mini",
        "model_provider" : "openai",
        'input_dir': 'uploads',
        'output_dir': 'ablation_results/annotation_context/',
        'h5ad_file': 'pbmc3k.h5ad',
        'markers_file': 'markers_pbmc3k.csv',
        'original_grouping': 'leiden',
        "output_column":"new_annotation",
        'initial_hypothesis': """
        """
    }
    
    pipeline = CellAnnotationPipeline(config)
    pipeline.run_pipeline(iterations=1)

    
#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from agents.hypothesis_agent.liver_hypothesis_agent import LiverHypothesisAgent
from agents.experiment_agent.liver_experiment_agent import LiverExperimentAgent
from agents.environment_agent.liver_environment_agent import LiverEnvironmentAgent
from agents.evaluation_agent.liver_evaluation_agent import LiverEvaluationAgent
from utils.liver_process_toolkit import solve_auto_fill_in, plot_2
import time

class CellAnnotationPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.initialize_directories()
        self.load_data()
        
        # Initialize agents
        self.hypothesis_agent: Optional[LiverHypothesisAgent] = None
        self.experiment_agent: Optional[LiverExperimentAgent] = None
        self.environment_agent: Optional[LiverEnvironmentAgent] = None
        self.evaluation_agent: Optional[LiverEvaluationAgent] = None
        
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
        
        # Generate markers if not provided
        markers_path = os.path.join(self.config['input_dir'], self.config['markers_file'])
        if not os.path.exists(markers_path):
            self.generate_markers_csv()

    def generate_markers_csv(self):
        """Generate marker genes CSV file"""
        sc.tl.rank_genes_groups(self.adata, self.config['original_grouping'], method='wilcoxon')
        marker_df = sc.get.rank_genes_groups_df(self.adata, group=None)
        marker_df = marker_df.rename(columns={
            'group': 'cluster',
            'names': 'gene',
            'logfoldchanges': 'avg_log2FC',
            'scores': 'score',
            'pvals_adj': 'p_val_adj'
        })
        marker_df.to_csv(os.path.join(self.config['input_dir'], self.config['markers_file']), index=False)

    def run_iteration(self):
        """Run one complete iteration of the pipeline"""
        # Hypothesis Stage
        self.hypothesis_agent = LiverHypothesisAgent(
            hypothesis=self.config['initial_hypothesis'],
            h5ad_file=os.path.join(self.config['input_dir'], self.config['h5ad_file']),
            csv_file=os.path.join(self.config['input_dir'], self.config['markers_file']),
            model_name = self.config['model_name'],model_provider = self.config['model_provider']
        )
        self.hypothesis_agent.identify_top_genes()
        refined_hypothesis = self.hypothesis_agent.refine_hypothesis(
            annotation_dict=self.annotation_dict, evaluation_result=self.evaluation, 
            no_gene_cluster=self.no_gene_cluster,iteration_summary=None
        )

        with open(self.config["log_file_path"], "a") as file:
            file.write(refined_hypothesis)
            file.write("\n")

        # Experiment Stage
        self.experiment_agent = LiverExperimentAgent(refined_hypothesis)
        if self.current_iteration == 0:
            self.experiment_agent.propose_experiment(model_name = self.config['model_name'],model_provider = self.config['model_provider'])
        else:
            self.experiment_agent.propose_experiment(
                self.annotation_dict, self.no_gene_cluster, self.failed_genes, self.successful_genes
            )
        marker_gene_proposal = self.experiment_agent.get_experiment_proposal()

        with open(self.config["log_file_path"], "a") as file:
            file.write(marker_gene_proposal)
            file.write("\n")
            

        # Environment Stage
        self.environment_agent = LiverEnvironmentAgent(
            self.config['input_dir'], self.config['output_dir'],
            self.config['h5ad_file'],
            model_name = self.config['model_name'],model_provider = self.config['model_provider']
        )

        marker_gene_list = self.environment_agent.extract_genes(marker_gene_proposal=marker_gene_proposal)
        existing_genes, dotplot = self.environment_agent.run_dotplot(
            marker_gene_list,
            self.current_iteration,
            groupby=self.config['original_grouping'],
            species = self.config['species']
        )

        # Evaluation Stage
        self.evaluation_agent = LiverEvaluationAgent(
            refined_hypothesis,self.config['output_dir'],self.config['input_dir'],
            existing_genes,self.config['h5ad_file'],
            model_name = self.config['model_name'],model_provider = self.config['model_provider']
        )

        start_time = time.time()
        evaluation_results = self.evaluation_agent.evaluate(
            groupby=self.config['original_grouping'],
            dotplot=dotplot,
            possible_cell_types=self.experiment_agent.get_experiment_proposal(),
            iteration=self.current_iteration
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Evaluation took {elapsed_time:.2f} seconds")

        evaluation, _,_,_,_,_ = evaluation_results
        with open(self.config["log_file_path"], "a") as file:
            file.write(str(elapsed_time)+" seconds\n")
            file.write(evaluation)
            file.write("\n")
           
        self.process_evaluation_results(*evaluation_results)
        self.current_iteration += 1

    def process_evaluation_results(self, evaluation, failed_genes, successful_genes, 
                                  marker_genes, empty_keys, similar_clusters_dict):
        """Process and store evaluation results"""
        self.failed_genes = failed_genes
        self.successful_genes = successful_genes
        self.evaluation = evaluation
        self.no_gene_cluster = self.evaluation_agent.find_no_gene_cluster(evaluation)
        prediction = self.evaluation_agent.prediction(evaluation)
        current_dict = self.evaluation_agent.execution(prediction)
        self.annotation_dict.update(current_dict)
        print("ANNOTATION:\n",self.annotation_dict)
        # Generate visualization
        self.generate_umap_visualization(current_iteration=self.current_iteration)
        self.save_results()

    def generate_umap_visualization(self,current_iteration):
        annotation_dict = self.annotation_dict
        groupby = self.config['original_grouping']
        if not "umap" in self.adata.uns:
            sc.pp.neighbors(self.adata)
            sc.tl.umap(self.adata)
        plt.figure(figsize=(10, 10))
        umap_filename = None
        org_dict = {int(i): str(i) for i in self.adata.obs[groupby]}
        org_dict.update(annotation_dict)
        output_column_name = self.config["output_column"]
        #self.adata.obs[groupby] = self.adata.obs[groupby].astype(int)
        self.adata.obs[output_column_name] = self.adata.obs[groupby].map(org_dict).astype('category')
        self.adata.obs[f"{groupby}_labels"] = self.adata.obs[groupby].map(org_dict).astype('category')
        sc.pl.umap(self.adata, color=f"{groupby}_labels", legend_loc='on data', title='UMAP plot', show=False)
        umap_filename = f"{current_iteration}_{self.config['h5ad_file']}_umap_plot.png"
        plt.savefig(os.path.join(self.config['output_dir'], umap_filename), dpi=300, bbox_inches='tight')
        plt.close()
        self.adata.obs[groupby] = self.adata.obs[groupby].astype("category")
        print('[DEBUG] UMAP plot saved as {}'.format(umap_filename))

    def save_results(self):
        """Save current state to files"""
        # Save AnnData object
        self.adata.write(os.path.join(self.config['input_dir'], self.config['h5ad_file']))
        
        # Save annotation dictionary
        pd.Series(self.annotation_dict).to_csv(
            os.path.join(self.config['output_dir'], f'annotations_iter_{self.current_iteration}.csv')
        )

    def run_pipeline(self, iterations: int = 3):
        """Run complete pipeline with specified number of iterations"""
        for _ in range(iterations):
            print(f"Current iteration {self.current_iteration}")
            self.run_iteration()
        if len(self.annotation_dict) * 2 < len(self.adata.obs[self.config["original_grouping"]].cat.categories):
            raise AttributeError("Annotation failed and will try again")

if __name__ == "__main__":
    ITERATION = 1
    config = {
        "model_name":"google/gemini-2.5-pro-preview",
        "model_provider" : "openrouter",
        'input_dir': 'uploads',
        'output_dir': 'outputs/',
        "log_file_path": "outputs/annotation_log_file.txt",
        'h5ad_file': 'retina.h5ad',
        'markers_file': 'markers_retina.csv',
        'original_grouping': 'leiden',
        "output_column":"new_annotation",
        'species':"human",
        'initial_hypothesis': """
    """
    }
    
    pipeline = CellAnnotationPipeline(config)
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pipeline.run_pipeline(iterations=ITERATION)
            break  # Success, so break the loop
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt < MAX_RETRIES:
                print("Retrying...")
            else:
                print("All retries failed.")

    #pipeline.run_pipeline(iterations=3)

    
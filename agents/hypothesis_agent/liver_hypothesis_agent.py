from agents.hypothesis_agent.base_hypothesis_agent import BaseHypothesisAgent
from utils.liver_process_toolkit import get_top_differential_genes
import scanpy as sc
from utils.LLM import query_llm

class LiverHypothesisAgent(BaseHypothesisAgent):
    def __init__(self, hypothesis, h5ad_file, csv_file=None,model_name = "gpt-4o",model_provider = "openai"):
        super().__init__(hypothesis)
        self.h5ad_file = h5ad_file
        if csv_file:
            self.marker_name = csv_file
        self.adata = sc.read_h5ad(self.h5ad_file)
        self.top_genes = None
        self.cluster_images = []
        self.iteration = 0
        self.reference_dict = None

        self.model_name = model_name
        self.model_provider = model_provider
        
    def identify_top_genes(self,type=None):
        n_genes = 5 if self.iteration == 0 else 3
        if type == "scanpy":
            self.top_genes = get_top_differential_genes(self.marker_name, n_genes=n_genes,cluster="group",foldchange="logfoldchanges",gene="names")
        else:
            self.top_genes = get_top_differential_genes(self.marker_name, n_genes=n_genes)
    
    def refine_hypothesis(self, annotation_dict=None,evaluation_result=None, no_gene_cluster=None,iteration_summary = None):
        self.iteration += 1

        if evaluation_result:
            # Filter top genes based on no_gene_cluster and failed_genes
            filtered_top_genes = {}
            for cluster, genes in self.top_genes.items():
                if cluster not in no_gene_cluster:
                    filtered_genes = [gene for gene in genes]
                    if filtered_genes:
                        filtered_top_genes[cluster] = filtered_genes
            self.top_genes = filtered_top_genes

        #content = f"Literature Summary:\n\n{self.summary}\n\n"
        content = f"Top {len(self.top_genes)} differentially expressed genes: {self.top_genes}\n\n"
        if self.reference_dict:
            content += f"You can refer to the possible cell types of these top genes in this dictionary{self.reference_dict}"
        content += f"Current Hypothesis:\n{self.hypothesis}\n\n"
        if annotation_dict:
            content += f"The cell type annotation from previous iterations {annotation_dict}"
        if no_gene_cluster:
            content += f"Clusters without need to be focused on: {no_gene_cluster}\n\n"
        #if evaluation_result:
        #    content += f"Evaluation Result:\n{evaluation_result}\n"
        if iteration_summary:
            content += f"This is summary of previous iteration annotation, with information of next steps to take. {iteration_summary}"

        system_role =  "You are a research assistant specializing in cell biology. Based on top differentially expressed genes, previous cell type annotation (if provided), Clusters without need to be focused on (if provided), summary of previous iteration annotation (if provided), and failed genes (if provided), refine the given hypothesis to be more accurate and specific."

        LLM_output = query_llm(content=content,system_role=system_role,model_provider=self.model_provider,model_name=self.model_name)

        self.refined_hypothesis = LLM_output
        return self.refined_hypothesis
        
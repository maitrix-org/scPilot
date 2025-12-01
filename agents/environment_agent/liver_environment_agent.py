import os
import re
import ast
import matplotlib
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from utils.LLM import query_llm
from agents.environment_agent.base_environment_agent import BaseEnvironmentAgent
matplotlib.use('Agg')

class LiverEnvironmentAgent(BaseEnvironmentAgent):
    def __init__(self, input_dir, output_dir, file_name,model_name = "gpt-4o",model_provider = "openai"):
        super().__init__(simulation_environment='liver', input_dir=input_dir, output_dir=output_dir)
        self.adata = ad.read_h5ad(os.path.join(input_dir, file_name))
        self.existing_genes = []  # Initialize the attribute

        self.model_name = model_name
        self.model_provider = model_provider
    
    def extract_genes(self, marker_gene_proposal):
        content = f"Here is input text:\n\n{marker_gene_proposal}\n\n"
        content += "Please extract the gene list at the end of input text, the input gene list will have format like this: MARKER_GENES = ['GeneA', 'GeneB', 'GeneC', ...]. You need to ONLY output the list ['GeneA', 'GeneB', 'GeneC', ...]. In mouse dataset, only capitalize first letter. for example, use the gene name 'Epcam' instead of 'EpCAM'. DO NOT INCLUDE genes such as 'H2-aa' because there is a dash inside."
        system_role =  "You are a research assistant processing text and look for gene list."
        LLM_output = query_llm(content=content,system_role=system_role,model_provider=self.model_provider,model_name=self.model_name)
        gene_list = LLM_output
        return gene_list
    
    def run_dotplot(self, marker_gene_list, iteration,groupby="seurat_clusters",species="mouse"):
        match = re.search(r"\[.*\]", marker_gene_list, re.DOTALL)
        if match:
            lst = ast.literal_eval(match.group())  # Convert string to list
        self.existing_genes = [gene for gene in lst if gene.lower() in map(str.lower, self.adata.var_names)]
        self.existing_genes = list(set(self.existing_genes))
        if self.existing_genes:
            if species == "human":
                self.existing_genes = [gene.upper() for gene in self.existing_genes]
            try:
                plt.figure(figsize=(20, 20))  # Increase figure size
                sc.pl.dotplot(self.adata, self.existing_genes, groupby=groupby, show=False)
                plt.tight_layout()
                dotplot_name = str(iteration)+'_01-marker_dotplot.png'
                plt.savefig(os.path.join(self.output_dir, dotplot_name), dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to free up memory
                dotplot = sc.pl.dotplot(self.adata, self.existing_genes, groupby=groupby, return_fig=True)
            except Exception as e:
                print(f"Error creating dotplot: {e}")
        else:
            print("No existing genes found. Skipping dotplot creation.")
        return self.existing_genes,dotplot
    
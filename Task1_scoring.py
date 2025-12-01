
import openai
from config.settings import OPENAI_API_KEY
from utils.liver_process_toolkit import plot_2, solve_rest_clusters
from owlready2 import get_ontology, Thing, Property

import pandas as pd  
import os
import re

# Set R_HOME environment variable if necessary
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from openai import OpenAI

pandas2ri.activate()

def get_cl_info(cell_types):
    # Convert the Python list to an R vector
    r_cell_types = ro.StrVector(cell_types)

    # R script to perform the ontology search
    r_script = """
    library(rols)

    find_cl_info <- function(cell_types) {
        # Initialize an empty list to store results
        results <- list()

        # Loop over each cell type in the list
        for (cell_type in cell_types) {
            d <- as(olsSearch(OlsSearch(q = cell_type, ontology = 'cl')), 'data.frame')
            d <- d[grep('CL:', d$obo_id),]

            # If a match is found, append to results
            if (nrow(d) > 0) {
                result_df <- data.frame(cell_type = cell_type, broadtype = d[1, 'label'], CLID = d[1, 'obo_id'], stringsAsFactors = FALSE)
            } else {
                result_df <- data.frame(cell_type = cell_type, broadtype = NA, CLID = NA, stringsAsFactors = FALSE)
            }

            results[[cell_type]] <- result_df
        }

        # Combine all results into a single dataframe
        cv <- do.call(rbind, results)
        return(cv)
    }

    # Call the function and return the result
    find_cl_info(cell_types)
    """

    # Set the R cell types list in the R environment
    ro.globalenv['cell_types'] = r_cell_types

    # Execute the R script
    result = ro.r(r_script)

    # Convert the R dataframe to a pandas dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        df = ro.conversion.rpy2py(result)

    return df

def reformat_list(lis):
    result_list = []
    for item in lis:
        parts = item.split(',')
        formatted_parts = [
            (part.strip().lower().replace("cells", "cell")[:-1]
             if part.strip().lower().replace("cells", "cell").endswith('s') and ' ' not in part.strip()
             else part.strip().lower().replace("cells", "cell"))
            for part in parts
        ]
        result_list.append(", ".join(formatted_parts))
    return result_list

def reformat_string(result):
    parts = result.split(',')
    formatted_parts = [
        (part.strip().lower().replace("cells", "cell")[:-1]
         if part.strip().lower().replace("cells", "cell").endswith('s') and ' ' not in part.strip()
         else part.strip().lower().replace("cells", "cell"))
        for part in parts
    ]
    return ", ".join(formatted_parts)

def map_cell_types(cell_group,cell_type_mapping):
    cells = cell_group.split(", ")
    mapped_cells = [reformat_string(cell_type_mapping.get(cell, cell)) for cell in cells]
    return ", ".join(mapped_cells)

def extract_unique_cell_types(annotations):
    unique_cell_types = set()
    for annotation in annotations:
        cell_types = [cell_type.strip() for cell_type in annotation.split(',')]
        unique_cell_types.update(cell_types)
    return list(unique_cell_types)

import scanpy as sc
def update_mapping(df_series, cell_type_mapping, batch_size=20):
    query_cell = []
    temp_mapping = {}
    # Identify cells not in the mapping
    for item in df_series:
        cells =  re.split(',', item)
        for cell in cells:
            cell = cell.strip()
            if cell not in cell_type_mapping.keys():
                query_cell.append(cell)
                cell_type_mapping[cell] = cell
    # Function to handle batch queries with retries for incomplete results
    def query_batch(batch_input):
        batch_message_content = "\n".join(batch_input)
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                                    {
                                        "role": "user",
                                        "content": (
                                            f"Please provide the full names for the following cell types. "
                                            "If a cell type contains a slash '/', keep it intact as a single unit. "
                                            "If the cell type is already a full name, simply repeat it. "
                                            "If it is not a valid cell name, simply repeat it."
                                            "Do not skip any of the cell types. "
                                            "Ensure the output matches the input order, with one full name per line. "
                                            "Only return the clean official names without any extra formatting, numbers, or symbols:\n"
                                            f"{batch_message_content}"
                                        )
                                    }
                                ],
            max_tokens=500,
            temperature=0.3,
        )
        return [name.strip() for name in response.choices[0].message.content.strip().split('\n') if name.strip()]
    # Process each batch and ensure completeness before moving to the next
    for i in range(0, len(query_cell), batch_size):
        batch_input = [cell for cell in query_cell[i:i + batch_size] if cell not in temp_mapping and cell.strip()]
        while True:
            batch_input = [cell for cell in batch_input if cell.strip()]
            batch_results = query_batch(batch_input)
            if len(batch_results) == len(batch_input):
                # If results are complete, map them and break the loop
                for original, official in zip(batch_input, batch_results):
                    temp_mapping[original] = official
                break
            else:
                continue
    # Update the main mapping once all cells are confirmed
    cell_type_mapping.update(temp_mapping)
    return cell_type_mapping

def process_data(h5ad_file='liver.h5ad',
                 marker_file='yan_markers.csv',
                 threshold=0.95,
                 tissue_name="",
                 cell_type_mapping=None,
                 cluster_column="seurat_clusters",
                 correct_column="celltype4_plotting",
                 output_column="cell_type_v1",
                 marker_cluster_column="cluster",
                 marker_filter_column="avg_log2FC",
                df_gpt=None,
                gene_column="gene",
                model_celltypist='Healthy_Mouse_Liver.pkl'):
    adata = sc.read_h5ad(h5ad_file)
    cluster_celltype_counts = adata.obs.groupby([cluster_column, correct_column], observed=False).size().unstack(fill_value=0)
    unique_celltypes_list = list(adata.obs[correct_column].unique())
    threshold_percentage = threshold
    filtered_celltypes = pd.DataFrame()
    for cluster in cluster_celltype_counts.index:
        current_data = cluster_celltype_counts.loc[cluster].sort_values(ascending=False)
        current_data.index = [x.replace('\n', '') for x in current_data.index]
        total_count = current_data.sum()
        threshold_value = total_count * threshold_percentage
        cumulative_sum = 0
        valid_celltypes = []
        for celltype, count in current_data.items():
            if cumulative_sum <= threshold_value:
                cumulative_sum += count
                valid_celltypes.append(celltype)
        filtered_celltypes.loc[cluster, 'CellTypes'] = ', '.join(valid_celltypes)
    cell_type_mapping = dict()
    cell_type_mapping['']=''
    cell_type_mapping=update_mapping(filtered_celltypes['CellTypes'], cell_type_mapping, batch_size=5)
    if cell_type_mapping:
        modified_cell_type_mapping = {key: reformat_string(value) for key, value in cell_type_mapping.items()}
        filtered_celltypes['CellTypes'] = filtered_celltypes['CellTypes'].apply(lambda x:map_cell_types(x,modified_cell_type_mapping))
    else:
        filtered_celltypes['CellTypes']=filtered_celltypes['CellTypes'].apply(lambda x:reformat_string(x))
    df_correct = get_cl_info(reformat_list(extract_unique_cell_types(filtered_celltypes['CellTypes'] )))
    temp=filtered_celltypes['CellTypes']
    results = []
    for annotation in temp:
        cell_types = annotation.split(', ')
        matching_rows = df_correct[df_correct['cell_type'].isin(cell_types)]
        # Create a single row per annotation with concatenated results
        if not matching_rows.empty:
                results.append({
                    'Annotation': annotation,
                    'cell_type': ", ".join(matching_rows['cell_type'].astype(str)),
                    'broadtype': ", ".join(matching_rows['broadtype'].astype(str)),
                    'CLID': ", ".join(matching_rows['CLID'].astype(str))
                })
    correct_df=pd.DataFrame(results)
    celltypist_df = correct_df
    results_df = correct_df
    cluster_celltype_counts = adata.obs.groupby([cluster_column, output_column], observed=False).size().unstack(fill_value=0)
    unique_celltypes_list = list(adata.obs[output_column].unique())
    ours_filtered_celltypes = pd.DataFrame()
    for cluster in cluster_celltype_counts.index:
        current_data = cluster_celltype_counts.loc[cluster].sort_values(ascending=False)
        current_data.index = [x.replace('\n', '') for x in current_data.index]
        total_count = current_data.sum()
        threshold_value = total_count * threshold_percentage
        cumulative_sum = 0
        valid_celltypes = []
        for celltype, count in current_data.items():
            if count==0:
                break
            if cumulative_sum <= threshold_value:
                cumulative_sum += count
                valid_celltypes.append(celltype)
        ours_filtered_celltypes.loc[cluster, 'CellTypes'] = ', '.join(valid_celltypes)
    ours_annotations=list(ours_filtered_celltypes["CellTypes"])
    ours_annotations=reformat_list(ours_annotations)
    ours_annotations = [re.sub(r'(?i)rod photoreceptor', 'rod', s) for s in ours_annotations]
    ours_annotations = [re.sub(r'(?i)müller glial', 'müller glia', s) for s in ours_annotations]
    ours_annotations = [re.sub(r'(?i)LSEC', 'Endothelial Cell', s) for s in ours_annotations]
    ours_annotations = [re.sub(r'(?i)Erythroid', 'Erythrocyte', s) for s in ours_annotations]
    ours_annotations = [re.sub(r'(?i)cytotoxic T', 'CD8 T', s) for s in ours_annotations]

    df = pd.DataFrame({"ours_annotations": ours_annotations})
    if cell_type_mapping:
        modified_cell_type_mapping = {key: reformat_string(value) for key, value in cell_type_mapping.items()}
        df["ours_annotations"] = df["ours_annotations"].apply(lambda x:map_cell_types(x,modified_cell_type_mapping))
    else:
        df["ours_annotations"]=df["ours_annotations"].apply(lambda x:reformat_string(x))
    ours_annotations = list(df["ours_annotations"])
    df_ours=get_cl_info(reformat_list(extract_unique_cell_types(ours_annotations)))
    results = []
    for annotation in list(ours_annotations):
        cell_types = annotation.split(', ')
        matching_rows = df_ours[df_ours['cell_type'].isin(cell_types)]
        # Create a single row per annotation with concatenated results
        if not matching_rows.empty:
            results.append({
                'Annotation': annotation,
                'cell_type': ", ".join(matching_rows['cell_type'].astype(str)),
                'broadtype': ", ".join(matching_rows['broadtype'].astype(str)),
                'CLID': ", ".join(matching_rows['CLID'].astype(str))
            })

    # Create a DataFrame from the results
    ours_results_df = pd.DataFrame(results)
    return correct_df,results_df,ours_results_df,celltypist_df
from owlready2 import get_ontology, Thing, Property
from collections import defaultdict, deque
def build_relation_graph(relation_df):
        parents_graph = defaultdict(set)
        children_graph = defaultdict(set)
        for _, row in relation_df.iterrows():
            parent, child = row['Parent'], row['Child']
            parents_graph[child].add(parent)
            children_graph[parent].add(child)
        return parents_graph, children_graph

def find_all_relatives(name, parents_graph, children_graph):
    relatives = set()
    parent_queue = deque([name])  
    child_queue=deque([name])  
    while parent_queue:
        current_parent = parent_queue.popleft()
        if current_parent in relatives:
            continue
        relatives.add(current_parent)
        if parents_graph[current_parent]:
            for parent in parents_graph[current_parent]:
                if parent not in relatives:
                    parent_queue.append(parent)
    while child_queue: 
        current_child= child_queue.popleft()   
        if current_child in relatives:
            continue
        relatives.add(current_child)
        if children_graph[current_child]:
            for child in children_graph[current_child]:
                if child not in relatives:
                    child_queue.append(child)
    return relatives

def calculate_agreement(df_correct, df_annotations,df_child_parent):
    parents_graph, children_graph = build_relation_graph(df_child_parent)
    def calculate_row_agreement(correct_row, annotation_row):
        correct_clid = set(correct_row['CLID'].split(', ')) if pd.notna(correct_row['CLID']) else set()
        annotation_clid = set(annotation_row['CLID'].split(', ')) if pd.notna(annotation_row['CLID']) else set()

        extended_annotation_clid = set()
        for clid in annotation_clid:
            extended_annotation_clid.update(find_all_relatives(clid.strip(), parents_graph, children_graph))
        extended_correct_clid=set()
        for clid in correct_clid:
            extended_correct_clid.update(find_all_relatives(clid.strip(), parents_graph, children_graph))

        partscore = len(correct_clid.intersection(extended_annotation_clid)) > 0 or len(annotation_clid.intersection(extended_correct_clid))>0
        fullscore = correct_clid == annotation_clid
        if 'malignant cell' in correct_row['broadtype'] and 'malignant cell' in correct_row['broadtype'] :
            fullscore = True

        # Return the agreement score
        if partscore and fullscore:
            return 1.0  # Full agreement
        elif partscore or fullscore:
            return 0.5  # Partial agreement
        else:
            return 0.0  # No agreement

    agreement_scores = []
    for i, (index, row) in enumerate(df_annotations.iterrows()):
        correct_row = df_correct.iloc[i]
        score = calculate_row_agreement(correct_row, row)
        agreement_scores.append(score)

    return agreement_scores

def get_score(correct,gpt,ours,celltypist):
    from collections import defaultdict, deque
    # Load the ontology
    ont = get_ontology("http://purl.obolibrary.org/obo/cl.owl").load()
    child_parent_relations = []
    # Iterate over each CL class
    for cls in ont.classes():
        if cls.name.startswith("CL"):
            # Handle subclass relationships to get Subclass pairs
            for subclass in cls.subclasses():
                child_parent_relations.append((subclass.name.replace("_",":"),cls.name.replace("_",":")))
    df_child_parent= pd.DataFrame(child_parent_relations, columns=['Child', 'Parent'])
    result1=calculate_agreement(correct,gpt,df_child_parent)
    result2=calculate_agreement(correct,ours,df_child_parent)
    result3=calculate_agreement(correct,celltypist,df_child_parent)
    gpt["agreement_score"]=result1
    gpt["correct_CLID"]=correct["CLID"]
    ours["agreement_score"]=result2
    ours["correct_CLID"]=correct["CLID"]
    celltypist["agreement_score"]=result3
    celltypist["correct_CLID"]=correct["CLID"]
    return df_child_parent


def all_in_one(h5ad_file='liver.h5ad',
                 marker_file='yan_markers.csv',
                 threshold=0.95,
                 tissue_name="",
                 cell_type_mapping=None,
                 cluster_column="seurat_clusters",
                 correct_column="celltype4_plotting",
                 output_column="cell_type_v1",
                 marker_column="cluster",
                 marker_filter_column="avg_log2FC",
                 gene_column="gene",
                 marker_cluster_column="cluster",
                 df_gpt=None,
                  model_celltypist='Healthy_Mouse_Liver.pkl'):
    correct,gpt,ours,celltypist=process_data(h5ad_file=h5ad_file,
                                  marker_file=marker_file,
                                  threshold=threshold,
                                  cell_type_mapping=None,
                                  cluster_column= cluster_column,
                                  correct_column=correct_column,
                                  output_column=output_column,
                                  df_gpt=df_gpt,
                                  tissue_name=tissue_name,
                                  marker_filter_column=marker_filter_column,
                                  gene_column=gene_column,
                                  marker_cluster_column=marker_cluster_column,
                                  model_celltypist=model_celltypist)
    get_score(correct,gpt,ours,celltypist)
    return correct,gpt,ours,celltypist

########
# changes made here
input_dir= "uploads/" 
h5ad_file = "retina.h5ad"
markers_file = 'markers_retina.csv'
original_grouping = "leiden"# "seurat_clusters"#
correct_column = "celltype"# "celltype"#
output_column ="new_annotation" #"seurat_clusters_labels"#
tissue_name = "human retina"
model_celltypist=""
threshold = 0.95
file_path = "outputs/cell_adv.txt"
detail_path = 'outputs/ours1.csv'
ground_truth_path = 'outputs/gpt1.csv'
########

correct1,gpt1,ours1,celltypist1=all_in_one(h5ad_file=os.path.join(input_dir, h5ad_file),cell_type_mapping=None,marker_file=os.path.join(input_dir, markers_file),cluster_column=original_grouping,correct_column=correct_column,tissue_name=tissue_name,model_celltypist=model_celltypist,output_column=output_column,threshold=threshold)

print("average score ours: ",ours1["agreement_score"].mean())
our_score = ours1["agreement_score"].mean()

score_string = str(our_score)#+","+str(gpt_score)+","+str(celltypist_score)
# Open the file in append mode and write the result
with open(file_path, 'a') as file:
    file.write(" "+score_string)
print(f"Results appended to {file_path}")
print(ours1,gpt1)
ours1.to_csv(detail_path, index=False)
gpt1.to_csv(ground_truth_path, index=False)

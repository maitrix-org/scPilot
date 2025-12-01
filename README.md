# scPilot[README.md](https://github.com/user-attachments/files/23861168/README.md)

# Running Guide
## set-up

### 1. API key
replace the OPENAI_API_KEY and GOOGLE_API_KEY in /config/settings.py with your own <be>.

### 2. env install
refer to requirements.txt for python environment. Note that you will need numpy < 2.0 to run py-Monocle.

### 3. large file download

you should download the files in our anonymous google drive:
https://drive.google.com/drive/folders/18AFRwp0eEftBgy2_WfQBtXrp39z4yn2w?usp=sharing

you should then put it into the scPilot folder and name it as uploads/

## running 3 tasks

### 0. LLM usage
currently we support 2 LLM API sources: openai and google. You should define it using the 'model_provider'. Then, choose a 'model_name' that exists, such as "gpt-4o".

### 1. Cell type annotation
run Task1_scPilot.py for scPilot version, run Task1_direct.py for direct prompting, and use Task1_scoring for scoring annotation.
Task 1 results are in Task1_results folder.

For each of the python files, please refer to its config section to specify the input / output folders.  

You should refer to CellTypeAnnotationDatasets.xlsx to fill in the configs related to dataset, such as original_grouping, correct_column, output_column, species and initial_hypothesis.

You can also change the LLM usage in the config section.

### 2. Trajectory inference
for 3 datasets used in trajectory inference task, run Traj_scPilot_1 to 3.ipynb for scPilot trajectory inference.
run Traj_direct_1 to 3.ipynb for direct trajectory inference.
Task 2 results are in Task2_results folder.

You can specify the LLM usage in the first notebook cell. Other than that, you can let itself run. In direct version notebooks, you may need to manually copy the tree from LLM response to ensure smooth running.

### 3. GRN TF-gene prediction
run Task3_combined.py, it will generate direct version and scPilot version prediction for the same questions in one run. 
Task 3 results are in Task3_results folder.

You can change the LLM usage in the config section.
If you want to try different tissues, you can change PREDICT_CONTEXT to "Liver" or "Stomach" or "Kidney".

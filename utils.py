# Load libraries
from os.path import join as pjoin
from os import listdir, getcwd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import FastICA
from statannotations.Annotator import Annotator
import json
import numpy as np
import os
import pandas as pd
import re 
import requests
import s3fs
import seaborn as sns


#Custom functions
def compute_stats(factors:pd.DataFrame, sample_metadata:pd.DataFrame) -> pd.DataFrame: 
    comparisons = [['Solid Tissue Normal', 'Primary Tumor'], 
                    ['Solid Tissue Normal', 'Metastatic'],
                    ['Primary Tumor', 'Metastatic']]

    full_df = factors.join(sample_metadata).reset_index().rename(columns  ={'index': 'sample_ids'})
    
    output = pd.DataFrame(columns= ['latent', 'variable', 'cor', 'pvalue'])

    for latent in factors.columns:
        print(latent)
        df_latent = full_df[['sample_ids', latent, 'sample_type']]
        df_melt = df_latent.melt(id_vars = [latent,'sample_ids'],
                                 var_name = 'metadata', 
                                 value_name = 'metadata_value')
       
        for combination in comparisons:
            u_statistic, p_value = mannwhitneyu(df_melt[df_melt.metadata_value == combination[0]][latent], 
                                                df_melt[df_melt.metadata_value == combination[1]][latent])
            
            
            output = pd.concat([output, 
                                pd.DataFrame({"latent": [latent], 
                                "variable": f'{combination[0]} v {combination[1]}',
                                "cor": u_statistic, 
                                "pvalue": p_value})])
                
    return output


def _download_tcga_data_files(gdc_manifest_path: str):
    """
    Function to download  RNAseq data from TCGA API
    
    Args: 
        gdc_manifest_path(str): path to gdc_manifest which has the relationship between folder 
            name and file name. Folder name must be in column A, file name must be in column B.
        
    Returns: downloads files
    """
    #Get file IDs from manifest
    file_ids = pd.read_csv(f'{gdc_manifest_path}', sep="\t", header=0).id.to_list()

    #Download data  through API
    data_endpt = "https://api.gdc.cancer.gov/data"
    params = {"ids": file_ids}
    response = requests.post(data_endpt,
                            data = json.dumps(params),
                            headers={
                                "Content-Type": "application/json"
                                })

    response_head_cd = response.headers["Content-Disposition"]

    file_name = re.findall("filename=(.+)", response_head_cd)[0]

    with open(file_name, "wb") as output_file:
        output_file.write(response.content)


def _download_tcga_metadata(primary_site: str):
    """"
    Function to download  metadata  from TCGA API
    
    Args: 
        primary_site(str): tissue type (e.g. "Breast", "Lung")

    Returns: formated pd.DataFrame with TCGA cases
    """
    #Download case data through TCGA API
    fields = [
        "id",
        "submitter_id",
        "case_id",
        "sample_id",
        "primary_site",
        "disease_type",
        "diagnoses.vital_status",
        "lost_to_followup",
        "days_to_lost_to_followup",
        "index",
        "state",
        "portion"]   

    fields = ",".join(fields)

    cases_endpt = "https://api.gdc.cancer.gov/cases"

    filters = {
        "op": "in",
        "content":{
            "field": "primary_site",
            "value": [f"{primary_site}"]
            }
        }

    params = {
        "filters": json.dumps(filters),
        "fields": fields,
        "format": "TSV",
        "size": "100"
        }

    response = requests.get(cases_endpt, params = params)
        
    #Format data into pd.DataFrame
    tcga_string = re.split('\t'+'\r', response.content.decode('UTF-8'))
    tcga_df = pd.DataFrame([sub.split("\r") for sub in tcga_string]).T
    tcga_df = tcga_df[0].str.split('\t', expand=True)

    tcga_df.columns = tcga_df.iloc[0]
    tcga_df.drop(tcga_df.index[0], inplace = True)

    tcga_df['case_id'] = tcga_df['case_id'].map(lambda x: x.lstrip('\n'))

    return tcga_df


def format_tcga_rnaseq(gdc_sample_sheet: str, root_dir: str,
                       workflow_type: str = 'unstranded'):
    """
    Function to format the RNAseq data from TCGA 

    Args:
        gdc_sample_sheet(str): path to tab separated gdc_sample_sheet file which has the folder Id ("File ID" column A), 
            "File Name" (column B) as well as the "Sample ID", "Case ID" etc.
        root_dir(str): path to the folder containing all the nested TCGA RNAseq data
        workflow_type(str): 'unstranded', 'stranded_first',	'stranded_second', 'tpm_unstranded',
            'fpkm_unstranded', 'fpkm_uq_unstranded'. 

    Returns:
        Formated pd.DataFrame
    """
    #Load sample sheet
    gdc_sample_sheet = pd.read_csv(f'{gdc_sample_sheet}', sep="\t", header=0)
    gdc_sample_sheet_rnaseq = gdc_sample_sheet[gdc_sample_sheet['Data Category'] == 'Transcriptome Profiling']

    #Join the files into a DataFrame, changing colname to sample_id
    _exists = s3fs.S3FileSystem().exists

    records = dict()
    for _, row in gdc_sample_sheet_rnaseq.iterrows():
        file_path = pjoin(root_dir, row["File ID"], row["File Name"])
        print(file_path)
        # if not _exists(file_path):
        #     print(f"WARNING: {file_path} doesn't exist")
        # else:
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=1).dropna()
        records[row["Sample ID"]] = df.set_index(["gene_id", "gene_name"])[f'{workflow_type}'].to_dict()

    tcga_df = pd.DataFrame(records)
    tcga_df.reset_index().rename(columns={"level_0": "gene_id", "level_1": "gene_name"}, 
                                 inplace= True)

    return tcga_df


def generate_boxplot(data: pd.DataFrame, x: str = 'sample_type', y: str = 'NES', 
                     figsize = (5, 5),
                     order: list = ['Solid Tissue Normal', 'Primary Tumor', 'Metastatic'],
                     title: str = 'miR-361-3p signature NES in TCGA-BRCA patients'):
    """

    """
    sns.set(rc = {'figure.figsize': figsize})
    sns.set(font_scale=1)
    plt = sns.boxplot(data=data, 
                    x=x, 
                    y=y, 
                    order= order )

    plt.set_title(title)


def load_transcriptomics(sample_metadata: pd.DataFrame,
                         source: str = 'gdc',
                         gdc_manifest_path: str = listdir(getcwd())):
    """
    Function to load transcriptomic data in workspace

    Args:
        source(str): 'gdc' or 'api'
        samples
        gdc_manifest_path

    Returns: pd.DataFrame with data ready to use
    """
    if source == 'gdc':
        tcga_df = pd.read_csv('EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', sep = '\t')
        #Format
        tcga_df['gene_id'] = tcga_df['gene_id'].str.split('|').str[0]
        tcga_df = tcga_df[~tcga_df.gene_id.str.startswith('?')].set_index('gene_id').rename(columns = lambda x: x[:-12])
        tcga_df = tcga_df[[col for col in tcga_df.columns if col in sample_metadata.index.to_list()]]
    
    elif source == 'api':
        if not [str(f).startswith('gdc_download_') for f in listdir(getcwd())]:
            _download_tcga_data_files(gdc_manifest_path)
        
        tcga_df = format_tcga_rnaseq(gdc_sample_sheet = 'gdc_sample_sheet.2023-07-11.tsv', 
                                     root_dir = 'gdc_download_20230711_113556.889465',
                                     workflow_type = 'tpm_unstranded')
        
    return  tcga_df   


def run_ica(n_components: int, data: pd.DataFrame):
    """
    Function to run independent component analysis

    Args:
        n_components(int): number of components
        data(pd.DataFrame): gex dataframe with samples as rows, genes (features) as columns

    Returns: loadings (gene weights), factors (patient weights)
    """
    ica = FastICA(n_components = n_components, 
                  random_state = 100, 
                  max_iter = 300)

    source_matrix = ica.fit_transform(data.T) # Reconstruct signals
    latent_ids = ['L'+ str(i) for i in range(n_components)]

    loadings = pd.DataFrame(source_matrix.T, 
                            columns=data.columns, 
                            index=latent_ids)

    factors = pd.DataFrame(ica.mixing_, 
                            index=data.index, 
                            columns=latent_ids)
    return loadings, factors
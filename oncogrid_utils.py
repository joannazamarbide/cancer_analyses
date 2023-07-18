# Load libraries
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns


# Custom functions
def convert_mutations_impact(mutation_data: pd.DataFrame, 
                             index = ['ModelID', 'HugoSymbol']):
    """
    Converts mutations into high, moderate, low or modifier impact. This classification is based
    on the Ensembl guidelines (link here: https://www.ensembl.org/info/genome/variation/prediction/predicted_data.html).
    
    Args:
        mutation_data(Dataframe): mutation dataframe. Columns must include ModelID (sample id), 
            HugoSymbol (gene name) and VariantInfo (consequence).
        index(list): List with 2 strings containing column names in mutation_data. These columns will 
            be used to drop duplicate rows. Defaults to 'ModelID' and 'HugoSymbol' to drop instances 
            where a single gene is mutated multiple times in the same sample. Because we sort by mutation impact, 
            the most high impact mutation will be retained. 
    
    Returns:
        mutation_impact_unique(Dataframe): DataFrame containing the mutation information in the gene of interest 
            for each sample, including the mutation_impact_score.
    """
    mutation_impact = mutation_data.copy()
    # create a list of our conditions according to: https://www.ensembl.org/info/genome/variation/prediction/predicted_data.html
    conditions = [
        (mutation_data['VariantInfo'] == 'SPLICE_SITE') |  (mutation_data['VariantInfo'] == 'NONSTOP')| 
            (mutation_data['VariantInfo'] == 'NONSENSE')| (mutation_data['VariantInfo'] == 'FRAME_SHIFT_INS') | 
            (mutation_data['VariantInfo'] == 'FRAME_SHIFT_DEL'),
        (mutation_data['VariantInfo'] == 'MISSENSE'),
        (mutation_data['VariantInfo'] == 'SILENT')]
    
    # Create a list of the values we want to assign for each condition
    values = [3, 2, 1]

    # Create a new column and use np.select to assign values to it using our lists as arguments
    mutation_impact['mutation_impact_score'] = np.select(conditions, values)

    # Sort values descending so the higher impact are higher up. 
    # then remove duplicates only keeping the high impact mutation for same gene of same patient.
    mutation_impact_unique = mutation_impact.sort_values('mutation_impact_score', ascending=False\
                                      ).drop_duplicates(subset=index, keep='first')

    # Replace the mutation_impact_score with high / medium / low 
    mutation_impact_unique['mutation_impact_score'] = mutation_impact_unique['mutation_impact_score'].replace([3, 2, 1], 
                                                 ['High', 'Moderate', 'Low'])
    
    return mutation_impact_unique


def compute_mannwhitney_stats(mutations_ic50_df: pd.DataFrame, 
                              split_by: str = 'HigMed_v_ModLowNon', 
                              n_min: int = 3.0):
    """
    Generates a stats table contaning mean IC50 of mutated and non-mutated groups, 
    p-value of difference in mean IC50 between groups, and number of cell lines in each group (mut vs non-mut). 
    
    Args:
        mutations_ic50_df (pd.DataFrame): Dataframe with the somatic mutations in genes of interest (genes_query) 
            for each cell lines, together with IC50 values.
        split_by (str): This can be 'HigMed_v_ModLowNon', 'HigMedMod_v_LowNon' or 'HigMedModLow_v_Non'. It denotes
            how to split the cell lines (e.g. split cell lines with high/medium mutations in gene X vs low/non)
        n_min (float): a threshold representing the minimum number of cell lines in either mutation to group to generate statistics. 
            Defaults to 3.0.
                        
    Returns:
        mannwhitneyu_output (pd.DataFrame): Statistics dataframe, which includes the gene name, indication, mean IC50 in each 
            group (Mut vs Non-mut), p-value, and number of cell lines in each group.
    """
    #Generate comparison groups:
    if split_by == 'HigMedModLow_v_Non':
        mutations_table = mutations_ic50_df.copy()
    elif split_by == 'HigMedMod_v_LowNon':
        mutations_table = mutations_ic50_df.copy()
        mutations_table['mutation_impact_score'] = mutations_table['mutation_impact_score'].replace('Low','not_mutated')
    elif split_by == 'HigMed_v_ModLowNon':
        mutations_table = mutations_ic50_df.copy()
        mutations_table['mutation_impact_score'] = mutations_table['mutation_impact_score'].replace('Low','not_mutated')

    #Make 2 mutation status groups by joining mutated under same category
    to_replace = ['High', 'Medium', 'Low'] 
    for i in to_replace:
        mutations_table['mutation_impact_score'] =mutations_table['mutation_impact_score'].replace(i,'mutated')
  
    
    #Generate empty DataFrame to store stats:
    mannwhitneyu_output = pd.DataFrame(columns = {'HugoSymbol', 'indication', 'mean_mutated', 
                                                  'mean_NOTmutated', 'mannwhitneyu_statistic', 
                                                  'p_value', 'n_cell_NOTmut', 'n_cell_mut'})
    
    #Compute stats:
    for gene in mutations_table['HugoSymbol'].unique():
        gene_df = mutations_table[mutations_table['HugoSymbol'] == gene]

        for tissue in mutations_table.indication.unique():
            cancer_df = gene_df[gene_df.indication == tissue]

            if (len(cancer_df[cancer_df.mutation_impact_score == 'not_mutated']['ic50']) >= n_min) \
                and (len(cancer_df[cancer_df.mutation_impact_score == 'mutated']['ic50']) >= n_min):                
                    mannwhitneyu_test = sp.stats.mannwhitneyu(list(cancer_df[cancer_df.mutation_impact_score == 'not_mutated']['ic50']),
                                                              list(cancer_df[cancer_df.mutation_impact_score == 'mutated']['ic50']))

                    mannwhitneyu_output = pd.concat([mannwhitneyu_output,
                                                     pd.DataFrame({'gene_name': gene,
                                                                   'indication': tissue,
                                                                   'mean_mutated': gene_df[gene_df.mutation_impact_score=='mutated']\
                                                                                           ['ic50'].mean(),
                                                                   'mean_NOTmutated': gene_df[gene_df.mutation_impact_score == 'not_mutated'] \
                                                                                           ['ic50'].mean(),
                                                                   'mannwhitneyu_statistic': pd.DataFrame.from_dict(mannwhitneyu_test).loc[0],
                                                                   'p_value': pd.DataFrame.from_dict(mannwhitneyu_test).loc[1],
                                                                   'n_cell_NOTmut': len(cancer_df[cancer_df.mutation_impact_score == 'not_mutated']\
                                                                                        ['depmap_id'].unique()),
                                                                   'n_cell_mut': len(cancer_df[cancer_df.mutation_impact_score == 'mutated']\
                                                                                     ['depmap_id'].unique()),
                                                                  }, index=[0])])
    #Reorder columns aesthetically
    mannwhitneyu_output = mannwhitneyu_output[['gene_name', 'indication', 'n_cell_mut', 'mean_mutated', 'n_cell_NOTmut', 
                                               'mean_NOTmutated', 'mannwhitneyu_statistic', 'p_value']]
                    
    return mannwhitneyu_output, mutations_table


def format_df_oncogrid(mutations_ic50_df: pd.DataFrame, 
                       sort_xaxis: str, 
                       mutation_column: str = 'mutation_impact_score', 
                       gene_name: str = 'HugoSymbol'):
    """
    Function to format the mutations DataFrame so it is suitable for plotting the oncogrid. 
    
    Args:
        mutations_ic50_df (pd.DataFrame): Dataframe with the somatic mutations in genes of interest (genes_query) 
            for each cell lines, together with IC50 values.
        sort_xaxis (str): This can be can be low_to_high_IC50, high_to_low_IC50, mutation_count. The plot x-axis (cell lines) 
            will be sorted according to this variable. If low_to_high_IC50/high_to_low_IC50: x-axis will be ordered from cell lines 
            with highest/lowest IC50 (left) to the reverse (right). If mutation_count, x-axis is ordered from cells with most 
            mutations (left) to least.
        mutation_column (str): column to use for heatmap values and color-coding (e.g. mutation_impact_score).
                        
    Returns:
        mut_table (pd.DataFrame): DataFrame with cell lines as columns and genes as rows. Values correspond to 
            mutation status/impact: 'High': 0, 'Low': 1, 'Moderate': 2, 'Modifier': 3, 'not_mutated': 4. This is 
            sorted based on sort_xaxis. 
        ic50_df (pd.DataFrame): DataFrame with cell lines and respective IC50 values, sorted based on sort_xaxis.
        value_to_int (Dict): Dictionary used to convert mutation status/impact values to int.     
    """
    #Refactor the mutation_column string values into int:
    value_to_int = {value: i for i, value in enumerate(sorted(pd.unique(mutations_ic50_df[mutation_column].ravel())))}
    sorted_mutation_data = mutations_ic50_df.copy()
    sorted_mutation_data[mutation_column] = sorted_mutation_data[mutation_column].replace(value_to_int)

    #Sort the mutation data table based on sort_xaxis 
    if sort_xaxis == 'low_to_high_IC50':
        sorted_mutation_data = sorted_mutation_data.sort_values(by = 'ic50', ascending=True)
        mut_table = sorted_mutation_data.copy().pivot(index=gene_name, 
                                                      columns='depmap_id', 
                                                      values=mutation_column) \
                                        .reindex(sorted_mutation_data.depmap_id.unique(), axis=1)
        ic50_df = sorted_mutation_data[['depmap_id', 'ic50', 'indication']].drop_duplicates()

    elif sort_xaxis == 'high_to_low_IC50':
        sorted_mutation_data = sorted_mutation_data.sort_values(by = 'ic50', ascending=False)
        mut_table = sorted_mutation_data.copy().pivot(index=gene_name, 
                                                      columns='depmap_id', 
                                                      values=mutation_column) \
                                        .reindex(sorted_mutation_data.depmap_id.unique(), axis=1)
        ic50_df = sorted_mutation_data[['depmap_id', 'ic50', 'indication']].drop_duplicates()

    elif sort_xaxis == 'mutation_count':
        cells_sorted_by_mutation = sorted_mutation_data.groupby('depmap_id')[mutation_column]\
                                   .sum().reset_index(name='count').sort_values(['count'], ascending=True).depmap_id.to_list()
        mut_table = sorted_mutation_data.copy().pivot(index=gene_name, 
                                                      columns='depmap_id', 
                                                      values=mutation_column).reindex(cells_sorted_by_mutation, axis=1)
        ic50_df = sorted_mutation_data[['depmap_id', 'ic50', 'indication']].drop_duplicates() \
                                    .set_index('depmap_id').reindex(cells_sorted_by_mutation, axis=0).reset_index()

    ## Reorder genes from most highly mutated to least
    mut_table = mut_table.apply(pd.to_numeric).assign(sum=mut_table.sum(axis=1)).sort_values(by='sum', ascending=True).iloc[:, :-1]

    return mut_table, ic50_df, value_to_int


def plot_oncogrid_ic50(mut_table: pd.DataFrame, ic50_df: pd.DataFrame, figsize: tuple, 
                       plot_title: str, label_fontsize: int, ticklabels_fontsize: int,
                       legend_fontsize: int, value_to_int: dict):
    """
    Plotting function that generates an oncogrid-type of plot with 2 subplots: a barplot showing IC50 for each cell line, 
    and a categorical heatmap showing mutation status/impact on each of the genes (y-axis) for each cell line (x-axis).
    
    Args:
        mut_table (DataFrame): sorted DataFrame with cell lines as columns and genes as rows. Values correspond to 
            mutation status/impact. 
        ic50_df (DataFrame): Sorted DataFrame with cell lines and respective IC50 values, sorted based on sort_xaxis.
        figsize: Size of figure. E.g. (20, 20).
        plot_title (str): Desired plot title.
        label_fontsize (int): Font size for plot labels.
        ticklabels_fontsize (int): Font size for plot tick-labels.
        legend_fontsize (int): Font size for legend.
        value_to_int (dict): Dictionary used to convert mutation status/impact values to int.     
                       
    Returns:
        plt: plot.
    """
    # Create the figure - getting the right height ratios can be tricky and often requires a trial-and-error process.
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 10]})
    [ax1, ax2] = axes
    fig.suptitle(plot_title, fontsize=20)

    # Create the IC50 bar plot.
    samples = list(mut_table.columns)
    sns.barplot(data=ic50_df, ax = ax1, x="depmap_id", y="ic50", 
               hue = 'indication', dodge=False, palette = "Set2")
    ax1.set_xlabel('')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylabel('IC50', fontsize=label_fontsize)
    ax1.tick_params(axis='y', which='major', labelsize=ticklabels_fontsize)

    # Create the categorical heatmap
    colors = ["#E50000", "#4374B3", "#F97306", "#7E1E9C", "#FFFFF0"]
   
    hm = sns.heatmap(mut_table.replace(value_to_int),
                     ax = ax2, cmap=colors, cbar=False,
                    linewidths=0.5, linecolor='black')
    # Add 1st legend
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    legend_ax = fig.add_axes([.7, .5, 1, .1])
    legend_ax.axis('off')
    
    # Add color map to legend
    patches = [mpatches.Patch(facecolor=c, edgecolor=c) for c in colors]

    # Add 2nd legend    
    leg1 = ax1.legend(patches,
                      sorted(value_to_int.keys()),
                      handlelength=0.8, 
                      title='Mutation impact', fontsize=legend_fontsize-2, title_fontsize=legend_fontsize, ncol = 2,  
                      bbox_to_anchor=(1, 0.1))

    ax1.add_artist(leg1)
    ax1.legend(title='indication', ncol = 2, bbox_to_anchor=(1, 1))

    # Plot!
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    
    return plt
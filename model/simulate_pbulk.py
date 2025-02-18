"""
Simulate pseudobulk for input groups of cells 
    groups: option1) from one single input domain; option2) the mixture of cells from all input domains 
Input: single cell (one batch or mixed multiple batches)
Output: simulated pseodubulk
"""

import scipy as sp
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
from copy import deepcopy 


def split_train_val_test_cells(sc_celltype_df, 
                               train_cells_ratio=0.8, val_cells_ratio=0.1): 
    
    """
    sc_celltype_df: [pd.DataFrame] cell_idx by cell_type
    """

    # sc_celltype_df = pd.DataFrame({"training_celltype": sc_adata_common.obs["celltype"].tolist()})
    ct_groups_dic = sc_celltype_df.groupby("training_celltype").groups

    cell_index_split_train = []
    cell_index_split_val = []
    cell_index_split_test = []

    for ct, cell_index in ct_groups_dic.items():
        print("----- {} -----".format(ct))
        num_selected_cells_train = int(train_cells_ratio * len(cell_index))
        num_selected_cells_val = int(val_cells_ratio * len(cell_index))
        num_selected_cells_test = int(len(cell_index) - num_selected_cells_train - num_selected_cells_val)
        print("Number of cells: train | val | test: {} | {} | {}".format(num_selected_cells_train, 
                                                                         num_selected_cells_val,
                                                                         num_selected_cells_test))

        cell_index_array = np.array(cell_index)
        np.random.shuffle(cell_index_array)
        cells_train = cell_index_array[:num_selected_cells_train]
        cells_val = cell_index_array[num_selected_cells_train:(num_selected_cells_train+num_selected_cells_val)]
        cells_test = cell_index_array[(num_selected_cells_train+num_selected_cells_val):]
        print("After split: train | val | test: {} | {} | {}".format(cells_train.shape[0], cells_val.shape[0], cells_test.shape[0]))
        cell_index_split_train.extend([i for i in cells_train])
        cell_index_split_val.extend([i for i in cells_val])
        cell_index_split_test.extend([i for i in cells_test])

    return cell_index_split_train, cell_index_split_val, cell_index_split_test




# generate training pseudo bulk data from single cell reference
def generate_pseudobulk(sc_data_df, sc_data_ct_df, 
                        celltype_level, desired_ct_lst,
                        outname=None,
                        samplenum=10, n_cells=500):
    """
    generate pseudobulk from single cell reference dataset
    :param sc_data_df: [pd.DataFrame] cell by genes; index is int array
    :param sc_data_ct_df: [pd.DataFrame] row - cell, column - different def for cell type groups
    :param celltype_level: [str] which level of cell type groups to use
    :param desired_ct_lst: [list] cell types to deconv; must cover all cell types in sc_data_ct_df (only allows that sc_data_ct_df missing cell types in desired_ct_lst.
    :param outname: [str] output filename and path
    :param samplenum: [int] how many pseudobulk samples for normal dirichlet dist
    :param n_cells: [int] number of cells in each pseudobulk sample
    :return: None
    """

    # sc_data_df : cell by genes
    sc_data = sc_data_df.copy()
    genename = sc_data.columns
    # choose cell type groups of interest
    num_celltype = len(set(sc_data_ct_df[celltype_level].unique()))
    celltype_groups_dic = sc_data_ct_df.groupby(celltype_level).groups
    celltypes_of_interest = sorted(list(celltype_groups_dic.keys()))

    print("Dirichlet dist with alpha being all ones, number of samples in total: {}".format(samplenum))
    prop = np.random.dirichlet(np.ones(num_celltype), samplenum)  # sampleNum by cellType

    # #### add more diversity in cell type proportions
    dSampleNum = ((samplenum // num_celltype) // 100) * 100
    if dSampleNum == 0:
        dSampleNum = ((samplenum // num_celltype) // 10) * 10
        if dSampleNum == 0:
            dSampleNum = 10
    # one cell type being dominant; no cell type prop = 0
    dominant_ct_prop = None
    for dct_i in range(num_celltype):
        alpha_lst = np.ones(num_celltype)
        alpha_lst[dct_i] = 50 # for 6 cell types it was 25
        if dct_i == 0:
            dominant_ct_prop = np.random.dirichlet(alpha_lst, dSampleNum)
        else:
            dominant_ct_prop = np.vstack((dominant_ct_prop, np.random.dirichlet(alpha_lst, dSampleNum)))
    print("Generate {} samples with one dominant cell type. Total sample number: {}".format(dSampleNum*num_celltype, dSampleNum*num_celltype + samplenum))
    # one cell type being around 60%
    mid_ct_prop = None
    for dct_i in range(num_celltype):
        alpha_lst = np.ones(num_celltype)
        alpha_lst[dct_i] = 15 # for 6 cell types it was 7
        if dct_i == 0:
            mid_ct_prop = np.random.dirichlet(alpha_lst, dSampleNum)
        else:
            mid_ct_prop = np.vstack((mid_ct_prop, np.random.dirichlet(alpha_lst, dSampleNum)))
    print("Generate {} samples with one cell type prop ~60%. Total sample number: {}".format(dSampleNum*num_celltype, 2*dSampleNum*num_celltype + samplenum))
    # one cell type being around 30%
    mid_ct_prop2 = None
    for dct_i in range(num_celltype):
        alpha_lst = np.ones(num_celltype)
        alpha_lst[dct_i] = 5 # for 6 cell types this 30% part does not exist
        if dct_i == 0:
            mid_ct_prop2 = np.random.dirichlet(alpha_lst, dSampleNum)
        else:
            mid_ct_prop2 = np.vstack((mid_ct_prop2, np.random.dirichlet(alpha_lst, dSampleNum)))
    print("Generate {} samples with one cell type prop ~30%. Total sample number: {}".format(dSampleNum*num_celltype, 3*dSampleNum*num_celltype + samplenum))



    prop = np.vstack((prop, dominant_ct_prop, mid_ct_prop, mid_ct_prop2))
    np.random.shuffle(prop)  # shuffle first dim in place

    cell_num = np.ceil(n_cells * prop)

    prop_df = pd.DataFrame(data = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1), columns=celltypes_of_interest)  # recalculate exact prop from number of cells
    sample = np.zeros((prop_df.shape[0], len(genename)))
    sample_celltype_specific_dic = {}
    for ct in celltypes_of_interest:
        sample_celltype_specific_dic[ct] = np.zeros((prop_df.shape[0], len(genename)))

    print('Sampling cells to compose pseudo-bulk data')
    for i, sample_prop in enumerate(cell_num):  # each sample
        for j, cellname in enumerate(celltypes_of_interest):  # each cell type
            select_index = np.random.choice(celltype_groups_dic[cellname], size=int(sample_prop[j]), replace=True)
            sum_selected_cells = sc_data.loc[select_index, :].sum(axis=0)
            if sample_prop[j] > 0:
                sample_celltype_specific_dic[cellname][i, :] = sum_selected_cells / int(sample_prop[j])  # avg; cell type specific GEP
                sample[i] += sum_selected_cells
    print('Sampling is done')

    sample_normalized = sample / np.sum(cell_num, axis=1).reshape(-1, 1)  # div by total number of cells in sampling

    # add missing cell type to prop_df & cell_num
    cell_num_df = pd.DataFrame(cell_num, columns=prop_df.columns, index=prop_df.index)
    if set(celltypes_of_interest) != set(desired_ct_lst):
        missing_ct = list(set(desired_ct_lst).difference(set(celltypes_of_interest)))
        for mct in missing_ct:
            prop_df[mct] = 0
            cell_num_df[mct] = 0

    # write data
    sample_adata = anndata.AnnData(X=sample_normalized) #, dtype=np.float64)
    sample_adata.obs_names = [str(i) for i in range(sample_adata.n_obs)]
    sample_adata.var_names = list(genename)
    sample_adata.uns["cellType_prop"] = prop_df
    sample_adata.uns["cellType_cellNum"] = cell_num_df
    for ct in celltypes_of_interest:
        sample_adata.layers[ct] = sample_celltype_specific_dic[ct]
    # add missing cell type to ctGEP
    for ct in desired_ct_lst:
        if ct not in sample_adata.layers:
            sample_adata.layers[ct] = np.zeros_like(sample_adata.X)
        else:
            continue

    if outname is not None:
        sample_adata.write_h5ad(outname + '.h5ad')
    else:
        return sample_adata

    print("Simulation DONE")



def simulate_pbulk(celltype_lst, out_save_path, save_filname, 
                   sc_count_df, sc_celltype_df, common_gene_lst,
                   cell_index_split_train_dic, cell_index_split_val_dic, cell_index_split_test_dic,
                   train_cells_ratio=0.8, val_cells_ratio=0.1,
                   train_samplenum=1000, val_samplenum=100, test_samplenum=100):
    """
    celltype_lst: [list] cell type of interest
    out_save_path: [str] path to save the simulated pbulk
    save_filename: [str] 
    sc_count_df: [pd.DataFrame] cells by genes

    """
    
    cell_index_split_train, cell_index_split_val, cell_index_split_test = split_train_val_test_cells(sc_celltype_df, train_cells_ratio, val_cells_ratio)
    cell_index_split_train_dic[save_filname] = deepcopy(cell_index_split_train)
    cell_index_split_val_dic[save_filname] = deepcopy(cell_index_split_val)
    cell_index_split_test_dic[save_filname] = deepcopy(cell_index_split_test)


    print("##### ----- {} Start ----- #####".format(save_filname))

    
    generate_pseudobulk(sc_count_df.loc[cell_index_split_train, common_gene_lst], 
                        sc_celltype_df.loc[cell_index_split_train, :], 
                        "training_celltype", 
                        desired_ct_lst=celltype_lst,
                        outname=out_save_path + save_filname + "_trainbulk",
                        samplenum=train_samplenum, n_cells=500)
    if val_samplenum > 0 and cell_index_split_val != []:
        generate_pseudobulk(sc_count_df.loc[cell_index_split_val, common_gene_lst], 
                            sc_celltype_df.loc[cell_index_split_val, :], 
                            "training_celltype", 
                            desired_ct_lst=celltype_lst,
                            outname=out_save_path + save_filname + "_valbulk",
                            samplenum=val_samplenum, n_cells=500)
    if test_samplenum > 0 and cell_index_split_test != []:
        generate_pseudobulk(sc_count_df.loc[cell_index_split_test, common_gene_lst], 
                            sc_celltype_df.loc[cell_index_split_test, :], 
                            "training_celltype", 
                            desired_ct_lst=celltype_lst,
                            outname=out_save_path + save_filname + "_testbulk",
                            samplenum=test_samplenum, n_cells=500)
    print("##### ----- {} DONE ----- #####".format(save_filname))
    print("\n\n")

    return cell_index_split_train_dic, cell_index_split_val_dic, cell_index_split_test_dic

import scipy as sp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



""" preprocessing bulk """

def preprocessing_bulk(count_data_df, desired_gene_order, up_scaling_coeff=1e6, plot_dist=False):
    count_data_df = count_data_df.loc[:, desired_gene_order]

    count_data = np.array(count_data_df.values, dtype=np.float64)
    count_data[count_data<0] = 0

    # lib size normalization
    count_data = count_data / np.sum(count_data, axis=-1).reshape(-1, 1)

    # scaling
    print("Up scaling by {}".format(up_scaling_coeff))
    count_data = count_data * up_scaling_coeff

    count_data = np.log2(count_data + 1)
    # normalize to [0,1]
    # count_data = count_data / np.max(count_data, axis=-1).reshape(-1, 1)

    # plot
    if plot_dist:
        # plt.title("lib size normalization + upscaling {} + log2 (+1) + mms".format(up_scaling_coeff))
        fig, axs = plt.subplots(1, 2)
        fig.set_figheight(4)
        fig.set_figwidth(8)
        axs[1].set_title("norm + upscale {} + log2".format(up_scaling_coeff))
        axs[1].hist(count_data.flatten(), log=True)

        axs[0].set_title("raw count")
        axs[0].hist(count_data_df.values.flatten(), log=True)
        plt.show()

    return count_data



""" Metric """
def L1Error(pred, true):
    return np.mean(np.abs(np.array(pred) - np.array(true)))


def CCCscore(y_pred, y_true):
    # input: one sample with multiple features: (size, )
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    # Pearson Product-moment Correlation Coefficients
    rho = np.corrcoef(y_pred, y_true)[0, 1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard Deviation
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * rho * std_true * std_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator


""" plot recon """
def plot_prop_pred_vs_gt(gt_pd, pred_pd, checkpoint_path, celltype_lst, num_prop, save_fig=False, filename="output1"):
    fig, axs = plt.subplots(nrows=1, ncols=num_prop)
    axs = axs.reshape(1, -1)
    fig.set_figheight(4)
    fig.set_figwidth(4*num_prop)
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    fig.supxlabel("gt")
    fig.supylabel("pred")
    i = 0

    for row_idx, ax in enumerate(axs):
        for j in range(len(ax)):
            if i < num_prop:
                ct = celltype_lst[i]
                x = gt_pd.loc[:, ct]
                y = pred_pd.loc[:, ct]

                l1 = L1Error(y, x)
                ccc = CCCscore(y, x)

                ax[j].scatter(x, y)
                ax[j].set_title(celltype_lst[i] + "\n l1 {:.4f} | ccc {:.3f}".format(l1, ccc))
                ax[j].plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="silver", linestyle="--")

            i += 1
    if save_fig:
        plt.savefig(checkpoint_path + filename + ".png")
    else:
        plt.show()
        plt.close()



from scipy.stats import gaussian_kde
def plot_ctGEP_pred_vs_gt_avg(gtGEP, predGEP, checkpoint_path, celltype_lst, num_prop, save_fig=False, filename="output3"):
    fig, axs = plt.subplots(nrows=1, ncols=num_prop)
    axs = axs.reshape(1, -1)
    fig.set_figheight(4)
    fig.set_figwidth(4*num_prop)
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    fig.supxlabel("gt")
    fig.supylabel("pred")
    i = 0

    for row_idx, ax in enumerate(axs):
        for j in range(len(ax)):
            x = np.mean(gtGEP[:, i, :], axis=0)
            y = np.mean(predGEP[:, i, :], axis=0)

            x_max = np.max(x)
            y_max = np.max(y)
            xy_max = max([x_max] + [y_max])

            xy = np.vstack([x,y])
            if x.any():
                z = gaussian_kde(xy)(xy)

                l1 = L1Error(y, x)
                ccc = CCCscore(y, x)

                ax[j].scatter(x, y, c=z)
                ax[j].set_xlim(-0.1, x_max+0.1)
                ax[j].set_ylim(-0.1, y_max+0.1)
                ax[j].set_title(celltype_lst[i] + "    " + "l1 {:.4f} | ccc {:.3f}".format(l1, ccc))
            elif not x.any():
                ax[j].set_title(celltype_lst[i] + "Missing in scref")
            ax[j].plot(np.linspace(0, xy_max, 100), np.linspace(0, xy_max, 100), color="silver", linestyle="--")

            i += 1
    if save_fig:
        plt.savefig(checkpoint_path + filename + ".png")
    else:
        plt.show()
        plt.close()

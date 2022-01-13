import matplotlib.pyplot as plt
import os
from os.path import join
import fnmatch
import numpy as np
import pickle
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(join(root, name))
    return result


def load_results(fname):
    data = []
    with open(fname, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    data = np.array(data)
    data = data.reshape([data.shape[0], data.shape[1]])
    return np.mean(data, axis=0), np.sqrt(np.var(data, axis=0))


def load_loss(fname):
    data = []
    with open(fname, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    data = np.array(data)
    return np.mean(data, axis=0), np.sqrt(np.var(data, axis=0))


def loss_plot(data_name, om_list, w0=-1, wa=0, save_fig=False, sim=False):
    nm, extent = os.path.splitext(data_name)

    # Import dataset
    data_path = find(data_name, current_dir)[0]
    if extent == '.txt':
        Data = pd.read_csv(data_path, usecols=[0, 1, 2], header=None, sep="   ", engine='python')
    if extent == '.csv':
        Data = pd.read_csv(data_path, header=0, sep=';', engine='python')
    Data = Data.to_numpy(dtype='float32')
    x = Data[:, 0].astype('float32')
    y = Data[:, 1].astype('float32')
    dy = Data[:, 2].astype('float32')

    H0 = 70E-6
    y = y / 5 + 1  # Ho y = log10(d) in pc
    y = y - np.log10(299792. / H0) - np.log10(1 + x)
    y = 10 ** y
    dy = y * dy
    order = x.argsort()

    if sim:
        file_dir = join(current_dir, 'simulations')
    else:
        file_dir = join(current_dir, 'omega_m')
    # Extracting data
    losses = np.array(
        [load_loss(find(nm + str(om) + '_loss.p', file_dir)[0]) for om in om_list])
    integrals = [np.array(load_results(find(nm + str(om) + '_integral.p', file_dir)[0])) for om in
                 om_list]
    bs = 30
    predictions = [np.array(load_results(find(nm + str(om) + '_prediction.p', file_dir)[0])) for om
                   in om_list]

    # Ground Truth
    confronto = (w0 + wa + 1) * np.log(1 + x) + wa / (1 + x) - wa

    # Plotting data
    col = ['red', 'green', 'blue', 'orange', 'black']  # plt.cm.get_cmap('Accent')
    fig, axs = plt.subplots(3, 1, figsize=(8, 15))

    for plt_ind in range(len(om_list)):
        axs[0].plot(x[order],
                    integrals[plt_ind][0][order],
                    label='$\Omega_m:$' + str(om_list[plt_ind]),
                    color=col[plt_ind])
        axs[0].fill_between(x[order],
                            integrals[plt_ind][0][order] - integrals[plt_ind][1][order],
                            integrals[plt_ind][0][order] + integrals[plt_ind][1][order],
                            alpha=.3,
                            color=col[plt_ind])

        axs[1].errorbar(x=x[order],
                        y=predictions[plt_ind][0][order],
                        yerr=predictions[plt_ind][1][order],
                        label='$\Omega_m:$' + str(om_list[plt_ind]),
                        color=col[plt_ind])

    axs[0].errorbar(x[order], confronto[order],
                    fmt='-.k',
                    linewidth=3,
                    label=f'$w_0$={w0} $w_a=${wa}',
                    zorder=0)
    axs[1].errorbar(x=x[order], y=y[order], yerr=dy[order],
                    markersize=4, fmt='ko',
                    ecolor='black', label='SN', zorder=0)

    # bicolore supernovae quasar
    # axs[1].errorbar(x=x[x < 2], y=y[x < 2], yerr=dy[x < 2],
    #                 markersize=3, fmt='bo', ecolor='green',
    #                 label='Q', zorder=0)

    axs[0].set_title('Integral $I_{NN}$')
    axs[1].set_title('Prediction')
    axs[2].set_title('Loss')
    axs[2].errorbar(x=om_list, y=losses[:, 0], yerr=losses[:, 1] / np.sqrt(bs), fmt='-bo', label='mse')

    axs[0].legend()
    axs[1].legend()
    fig.suptitle(data_name)
    plt.tight_layout()
    if save_fig:
        fig_dir = join(current_dir, 'Article', 'Figures')
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(join(fig_dir, 'full_plot' + nm + '.png'))

    plt.show()


def loss_plot_article(data_name, om_list, w0=-1, wa=0, save_fig=False, sim=False):
    nm, extent = os.path.splitext(data_name)

    # Import dataset
    data_path = find(data_name, current_dir)[0]
    if extent == '.txt':
        Data = pd.read_csv(data_path, usecols=[0, 1, 2], header=None, sep="   ", engine='python')
    if extent == '.csv':
        Data = pd.read_csv(data_path, header=0, sep=';', engine='python')
    Data = Data.to_numpy(dtype='float32')
    x = Data[:, 0].astype('float32')
    y = Data[:, 1].astype('float32')
    dy = Data[:, 2].astype('float32')

    H0 = 70E-6
    y = y / 5 + 1  # Ho y = log10(d) in pc
    y = y - np.log10(299792. / H0) - np.log10(1 + x)
    y = 10 ** y
    dy = y * dy
    order = x.argsort()

    if sim:
        file_dir = join(current_dir, 'simulations')
    else:
        file_dir = join(current_dir, 'omega_m')
    # Extracting data
    losses = np.array(
        [load_loss(find(nm + str(om) + '_loss.p', file_dir)[0]) for om in om_list])
    integrals = [np.array(load_results(find(nm + str(om) + '_integral.p', file_dir)[0])) for om in
                 om_list]
    bs = 30
    predictions = [np.array(load_results(find(nm + str(om) + '_prediction.p', file_dir)[0])) for om
                   in om_list]

    # Ground Truth
    confronto = (w0 + wa + 1) * np.log(1 + x) + wa / (1 + x) - wa

    # Plotting data
    col = ['red', 'green', 'blue', 'orange', 'black']

    # Plot Integral
    fig0, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.set_title('Integral $I_{NN}$')

    for plt_ind in range(len(om_list)):
        ax.plot(x[order],
                integrals[plt_ind][0][order],
                label='$\Omega_m:$' + str(om_list[plt_ind]),
                color=col[plt_ind])
        ax.fill_between(x[order],
                        integrals[plt_ind][0][order] - integrals[plt_ind][1][order],
                        integrals[plt_ind][0][order] + integrals[plt_ind][1][order],
                        alpha=.3,
                        color=col[plt_ind])

    ax.plot(x[order], confronto[order],
            '-.k',
            linewidth=3,
            label=f'$w_0$={w0} $w_a=${wa}',
            zorder=0)
    ax.legend(loc='upper left')
    if save_fig:
        fig_dir = join(current_dir, 'Article', nm)
        os.makedirs(fig_dir, exist_ok=True)
        fig0.savefig(join(fig_dir, 'integral' + nm + '.png'))
    ax.set_xlabel('z')
    plt.tight_layout()
    plt.show()

    # Plot prediction
    fig1, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.set_title('Hubble diagram $y_{NN}$')
    for plt_ind in range(len(om_list)):
        ax.plot(x[order],
                predictions[plt_ind][0][order],
                label='$\Omega_m:$' + str(om_list[plt_ind]),
                color=col[plt_ind])
        ax.fill_between(x[order],
                        predictions[plt_ind][0][order] - predictions[plt_ind][1][order],
                        predictions[plt_ind][0][order] + predictions[plt_ind][1][order],
                        alpha=.3,
                        color=col[plt_ind])
    ax.errorbar(x=x[order],
                y=y[order],
                yerr=dy[order],
                markersize=4,
                fmt='ko',
                ecolor='black',
                label='SN',
                zorder=0)
    ax.legend(loc='lower right')
    ax.set_xlabel('z')
    if save_fig:
        fig_dir = join(current_dir, 'Article', nm)
        os.makedirs(fig_dir, exist_ok=True)
        fig1.savefig(join(fig_dir, 'prediction' + nm + '.png'))
    plt.show()

    fig2, ax = plt.subplots(1, 1, figsize=(5, 2))
    ax.set_title('Mean Loss')
    ax.errorbar(x=om_list, y=losses[:, 0], yerr=losses[:, 1] / np.sqrt(bs), fmt='-bo')
    ax.set_xticks(om_list)
    ax.set_xlabel('$\Omega_m$')
    plt.tight_layout()
    if save_fig:
        fig_dir = join(current_dir, 'Article', nm)
        os.makedirs(fig_dir, exist_ok=True)
        fig2.savefig(join(fig_dir, 'loss' + nm + '.png'))
    plt.show()


if __name__ == "__main__":
    # for ds in ['dataset_A0.01.txt',
    #            # 'dataset_B1.5_0.01.txt',
    #            'dataset_C0.14.txt']:
    #            # 'dataset_D1.5_0.14.txt',
    #            # 'dataset_F1.5_0.14.txt']:
    #     loss_plot(ds,
    #               [0.2, 0.3, 0.4],  # [0.3, 0.15, 0.2, 0.3, 0.4, 0.45],
    #               w0=-1, wa=0,
    #               sim=False,
    #               save_fig=True)

    loss_plot_article('Real_Data.csv',
                      [0.2, 0.3, 0.4],  # [0.3, 0.15, 0.2, 0.3, 0.4, 0.45],
                      w0=-1, wa=0,
                      sim=False,
                      save_fig=True)

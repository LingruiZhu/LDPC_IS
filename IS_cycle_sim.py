import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import commpy.channelcoding as cc

from LDPC import encoder, decoder
from channel import awgn_IS, awgn_MC
from evaluation import fer_MC, fer_IS, ber_MC, ber_IS


def IS_cycle_sim(cycle_nodes_file, code_alist, snrs, n_trials, plot):

    # get parameters of ldpc
    paras = cc.get_ldpc_code_params(code_alist, compute_matrix=True)
    n_bit = 48
    message =  np.zeros([1, n_bit])

    # get cycle nodes
    cycle_nodes = pd.read_csv(cycle_nodes_file, index_col=0)    # data type: DataFrame
    cycle_rows = cycle_nodes.shape[0]

    # initializazion of bers and fers
    bers_mc_mat = -1 * np.ones([len(snrs), n_trials, cycle_rows])
    bers_is_mat = -1 * np.ones([len(snrs), n_trials, cycle_rows])
    fers_mc_mat = -1 * np.ones([len(snrs), n_trials, cycle_rows])
    fers_is_mat = -1 * np.ones([len(snrs), n_trials, cycle_rows])

    for snr in snrs:
        for j in range(n_trials):
            for i in range(cycle_rows):
                print('current snr: %d, current trails, %d current rows: %d' %(snr, j, i))
                bias_bits = np.array(cycle_nodes.iloc[0, :])
                bias_bits = bias_bits[np.where(~np.isnan(bias_bits))]
                bias_bits = bias_bits.tolist()
                bias_bits = np.linspace(0, 5, 6)


                # encoding and mapping
                c = encoder(message, paras)
                x = pow(-1, c)

                # awgn channel
                y_IS, weights = awgn_IS(x, snr, bias_bits, mode='VS')
                y_MC = awgn_MC(x, snr)

                # decoding
                c_hat_IS = decoder(y_IS, paras, max_iter=20)
                c_hat_MC = decoder(y_MC, paras, max_iter=20)

                fers_mc_mat[snr, j, i], _ = fer_MC(c_hat_MC, c)
                fers_is_mat[snr, j, i], _ = fer_IS(c_hat_IS, c, weights)
                bers_mc_mat[snr, j, i] = ber_MC(c_hat_MC, c)
                bers_is_mat[snr, j, i] = ber_IS(c_hat_IS, c, weights)

    fers_MC_snr = np.mean(fers_mc_mat, axis=(1, 2))
    fers_IS_snr = np.mean(fers_is_mat, axis=(1, 2))
    bers_MC_snr = np.mean(bers_mc_mat, axis=(1, 2))
    bers_IS_snr = np.mean(bers_is_mat, axis=(1, 2))

    cv_fers_MC_snr = np.std(np.mean(fers_mc_mat, axis=2), axis=1) / fers_MC_snr
    cv_fers_IS_snr = np.std(np.mean(fers_is_mat, axis=2), axis=1) / fers_IS_snr
    cv_bers_MC_snr = np.std(np.mean(bers_mc_mat, axis=2), axis=1) / bers_MC_snr
    cv_bers_IS_snr = np.std(np.mean(bers_is_mat, axis=2), axis=1) / fers_IS_snr

    print('coefficient of variation of FERS:')
    print(cv_fers_MC_snr)
    print(cv_fers_IS_snr)

    print('coefficient of variation of BERS:')
    print(cv_bers_MC_snr)
    print(cv_bers_IS_snr)


    # plot the result
    if plot:
        plt.figure()
        plt.semilogy(snrs, fers_MC_snr, color='red', label='MC')
        plt.semilogy(snrs, fers_IS_snr, color='blue', label='IS')
        plt.xlabel('SNR')
        plt.ylabel('FER')
        plt.title('FER vs SNR')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.semilogy(snrs, bers_MC_snr, color='red', label='MC')
        plt.semilogy(snrs, bers_IS_snr, color='blue', label='IS')
        plt.xlabel('SNR')
        plt.ylabel('BER')
        plt.title('BER vs SNR')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.semilogy(snrs, cv_bers_MC_snr, color='red', label='MC')
        plt.semilogy(snrs, cv_bers_IS_snr, color='blue', label='IS')
        plt.xlabel('SNR')
        plt.ylabel('Coefficient of Variation')
        plt.title('CV vs SNR')
        plt.legend()
        plt.grid()

        plt.show()

    return fers_IS_snr, fers_MC_snr, bers_IS_snr, bers_MC_snr


def mc_sim(snrs, code_alist):
    paras = cc.get_ldpc_code_params(code_alist, compute_matrix=True)
    n_bits = 48
    n_trials = 2000
    message = np.zeros([n_trials, n_bits])
    c = encoder(message, paras)
    x = pow(-1, c)
    sim_nums = []
    bers = []

    for snr in snrs:
        y = awgn_MC(x, snr)
        c_hat = decoder(y, paras)
        ber = np.mean(c_hat)
        std = np.std(np.mean(c_hat, axis=0))
        cv = std/ber
        i = 0
        while cv > 0.03:
            i = i+1
            print('current snr: %d, current loop: %d' %(snr, i))
            x_new = np.ones([50, 96])
            y_new = awgn_MC(x_new, snr)
            c_hat_new = decoder(y_new, paras)
            c_hat = np.concatenate([c_hat, c_hat_new], axis=0)
            ber = np.mean(c_hat)
            std = np.std(np.mean(c_hat, axis=0))
            cv = std / ber
            print('current variation coefficient %f' %cv)
            print('current codewordnum: %d' %c_hat.shape[0])

        sim_num = n_trials + i*50
        sim_nums.append(sim_num)
        bers.append(ber)

    return sim_nums, ber

if __name__ == '__main__':
    code_para_file = 'code_matrix/96_3_963.txt'
    cycle_file = 'cycles_info/cycles.csv'
    snrs_mc = np.linspace(6, 8, 3)
    mc_nums, mc_bers = mc_sim(snrs_mc, code_para_file)
    print(mc_nums)
    print(mc_bers)
    input('here to pause')



    snrs = np.linspace(0, 15, 16, dtype=int)
    n_trails = 20
    fers_IS, fers_MC, bers_IS, bers_MC = IS_cycle_sim(cycle_file, code_para_file, snrs, n_trails, True)

    # snr_df = pd.DataFrame({'FER_IS': fers_IS, 'BER_IS': bers_IS, 'FER_MC': fers_MC, 'BER_MC': bers_MC})
    # save_file = 'result_data/result.csv'
    # snr_df.to_csv(save_file)

    # save data to file
    n_loops = n_trails * 96
    snr_max = max(snrs)
    ber_IS_save_file = 'bers_fers_folder/cycle_IS/rBER_IS_snr_' + str(int(snr_max)) + '_dB_' + str(n_loops) + '_loops.npy'
    ber_MC_save_file = 'bers_fers_folder/cycle_IS/rBER_MC_snr_' + str(int(snr_max)) + '_dB_' + str(n_loops) + '_loops.npy'
    fer_IS_save_file = 'bers_fers_folder/cycle_IS/rFER_IS_snr_' + str(int(snr_max)) + '_dB_' + str(n_loops) + '_loops.npy'
    fer_MC_save_file = 'bers_fers_folder/cycle_IS/rFER_MC_snr_' + str(int(snr_max)) + '_dB_' + str(n_loops) + '_loops.npy'
    fer_CV_MC_save_file = 'bers_fers_folder/cycle_IS/rCV_FER_MC_snr_' + str(int(snr_max)) + '_dB_' + str(n_loops) + '_loops.npy'
    fer_CV_IS_save_file = 'bers_fers_folder/cycle_IS/rCV_FER_IS_snr_' + str(int(snr_max)) + '_dB_' + str(n_loops) + '_loops.npy'


    np.save(ber_IS_save_file, np.array(bers_IS))
    np.save(ber_MC_save_file, np.array(bers_MC))
    np.save(fer_IS_save_file, np.array(fers_IS))
    np.save(fer_MC_save_file, np.array(fers_MC))
    # np.save(fer_CV_IS_save_file, np.array(cv_IS))
    # np.save(fer_CV_MC_save_file, np.array(cv_MC))

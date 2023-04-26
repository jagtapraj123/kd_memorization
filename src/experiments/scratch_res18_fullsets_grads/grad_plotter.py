import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # res18
    # gradfile = "logs/logfiles/train_scratch_res18_0-1_grads_20230410-221525/grad_log2.pkl"
    
    # res18_reduced_by2
    # gradfile = "logs/logfiles/train_scratch_res18_reduced_by2_0-1_grads_20230410-221525/grad_log2.pkl"
    
    # res18_reduced_by4
    # gradfile = "logs/logfiles/train_scratch_res18_reduced_by4_0-1_grads_20230410-221525/grad_log2.pkl"
    
    # res18_reduced_by8
    gradfile = "logs/logfiles/train_scratch_res18_reduced_by8_0-1_grads_20230410-221525/grad_log2.pkl"

    with open(gradfile, "rb") as f:
        data = pickle.load(f)

    print(data.shape)

    infl = np.load("../../../dataset/cifar-100-python/cifar100_infl_matrix.npz")
    tr_mem = infl['tr_mem'].copy()
    
    print(tr_mem.shape)

    grad_bins = np.zeros((100, 200))
    grad_bin_cts = np.zeros((100,))

    # 0.01 * m, m>0, m<=100
    for i in range(tr_mem.shape[0]):
        m_bin = int(tr_mem[i]//0.01)
        
        grad_bins[m_bin, :] += data[i, :]
        grad_bin_cts[m_bin] += 1

    for m_bin in range(100):
        grad_bins[m_bin] /= grad_bin_cts[m_bin]

    print(grad_bins[0])
    print(grad_bins[99])

    # for m in [0, 99]:
    for m in range(0, 100):
        plt.plot(grad_bins[m], label="{}-{}".format(0.01*m, 0.01*(m+1)), color = [m/99, 1-m/99, 0])
        # plt.plot(grad_bins[99])
    # plt.legend()
    plt.savefig("train_scratch_res18_reduced_by8_0-1_grads_2.png")

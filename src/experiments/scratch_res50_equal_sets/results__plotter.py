import matplotlib.pyplot as plt
import os
import re

if __name__ == "__main__":
    mem_thresholds = [0.1, 0.2, 0.4, 0.6, 0.8, 1]

    file_prefix = "runlog_scratch_res50_equal_sets_0-{}.log"
    n = 3
    test_names = ["Training", "Full Train Set", "Test Set"]

    score_lists = [[] for _ in range(n)]
    for t in mem_thresholds:
        logfile = file_prefix.format(t)
        with open(logfile, 'rb') as f:
            num_newlines = 0
            try:  # catch OSError in case of a one line file 
                f.seek(-2, os.SEEK_END)
                print(f.tell())
                while num_newlines < n:
                    f.seek(-2, os.SEEK_CUR)
                    if f.read(1) == b'\n':
                        num_newlines += 1
            except OSError:
                f.seek(0)
            print(f.tell())
            for i in range(n):
                line = f.readline().decode()
                x = re.search("{'f1_score_macro': ([0-9].[0-9]+)}", line).group(1)
                print(x)
                score_lists[i].append(float(x))

    print(score_lists)
    for i in range(n):
        plt.plot(mem_thresholds, score_lists[i], marker='.', label=test_names[i])

    plt.xlim((0, 1.1))
    plt.xlabel("Memorization Threshold")
    plt.ylabel("F1 Score (macro)")

    plt.legend()
    plt.savefig("f1_macro.pdf")
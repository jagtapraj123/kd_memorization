import sys
sys.path.append("../../")

from utils.cifar_subset_creater import make_disjoint
import argparse
import pandas as pd
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Subset Generator", description="Generate Disjoint Subsets of CIFAR dataset")
    parser.add_argument("dir_path", type=str, help='Directory Path to store subset files (without spaces)')
    parser.add_argument("--seed", dest="seed", type=int, help="Random seed to select set1")

    args = parser.parse_args()

    dir_path = args.dir_path
    seed = args.seed
    
    print("Dir Name: {}".format(dir_path))

    set1, set2 = make_disjoint(list(range(50000)), seed)

    os.makedirs(dir_path, exist_ok=True)

    set1_df = pd.DataFrame(set1, columns=["idx"])
    set1_df.to_csv(os.path.join(dir_path, "subset_0-1_S1.csv"), index=False)

    set2_df = pd.DataFrame(set2, columns=["idx"])
    set2_df.to_csv(os.path.join(dir_path, "subset_0-1_S2.csv"), index=False)
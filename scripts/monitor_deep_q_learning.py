import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id_suffix", type=str, help="id of running training process", default="default")
    parser.add_argument('--summary_file', required=True)
    args = parser.parse_args()
    df_summary = pd.read_pickle(os.path.abspath(args.summary_file))
    df_summary.plot(subplots=True)
    plt.show()


if __name__ == '__main__':
    main()

import argparse

from arc_tigers.eval.utils import get_metric_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sampling metrics.")
    parser.add_argument(
        "data_dir", type=str, help="Directory containing the metrics files."
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    get_metric_stats(data_dir)

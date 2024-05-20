import argparse
import os
import json
from adapt_embed.utils import plot_comparison


def main():
    parser = argparse.ArgumentParser(description="Plot comparison graphs from JSON results.",
                                     epilog="Example: python plotting.py --save-dir plots path/to/results1.json Title1 path/to/results2.json Title2")
    parser.add_argument('--save-dir', type=str, required=True, help='Directory to save the plots.')
    parser.add_argument('--exp-name', type=str, default="", help='Experiment name, used for plot titles.')
    parser.add_argument('results', nargs='+', help='Pairs of path_to_json and title, e.g., path/to/results.json Title')
    
    args = parser.parse_args()
    
    if len(args.results) % 2 != 0:
        raise ValueError("Results arguments should be in pairs of path_to_json and title.")
    
    results_with_names = []
    split = None
    for i in range(0, len(args.results), 2):
        path_to_json, title = args.results[i:i+2]
        if not os.path.isfile(path_to_json):
            raise FileNotFoundError(f"{path_to_json} does not exist.")
        with open(path_to_json, 'r') as f:
            results = {"default": json.load(f)}
            if split is None:  # auto-detect split
                for k, v in results["default"].items():
                    if isinstance(v, dict):
                        split = k
                        break
        results_with_names.append((results, title))
    
    plot_comparison(results_with_names, save_dir=args.save_dir, exp_name=args.exp_name, split=split)


if __name__ == "__main__":
    main()

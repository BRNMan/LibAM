import argparse
import csv
import json
import os
from collections import defaultdict

from settings import DATA_PATH


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_final_scores(payload):
    """Recursively extract all final_score values from nested JSON payload."""
    scores = []

    def walk(node):
        if isinstance(node, dict):
            if "final_score" in node:
                score = _to_float(node["final_score"])
                if score is not None:
                    scores.append(score)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(payload)
    return scores


def _parse_pair_from_filename(filename):
    stem = filename
    if stem.endswith("_feature_result.json"):
        stem = stem[: -len("_feature_result.json")]
    parts = stem.split("___", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def build_pair_scores(area_dir):
    pair_scores = defaultdict(list)
    all_binaries = set()

    for name in sorted(os.listdir(area_dir)):
        if not name.endswith("_feature_result.json"):
            continue

        object_name, candidate_name = _parse_pair_from_filename(name)
        if object_name is None:
            continue

        all_binaries.add(object_name)
        all_binaries.add(candidate_name)

        path = os.path.join(area_dir, name)
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        scores = _extract_final_scores(payload)
        if scores:
            pair_scores[(object_name, candidate_name)].extend(scores)

    return pair_scores, sorted(all_binaries)


def build_matrix(pair_scores, binaries):
    avg_scores = {}
    for pair, scores in pair_scores.items():
        avg_scores[pair] = sum(scores) / float(len(scores))

    matrix = []
    for row_bin in binaries:
        row = []
        for col_bin in binaries:
            row.append(avg_scores.get((row_bin, col_bin), 0.0))
        matrix.append(row)
    return matrix, avg_scores


def write_csv(output_csv, binaries, matrix):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["binary"] + binaries)
        for name, row in zip(binaries, matrix):
            writer.writerow([name] + [f"{v:.6f}" for v in row])


def write_json(output_json, binaries, matrix, avg_scores):
    data = {
        "binaries": binaries,
        "matrix": matrix,
        "pair_average_scores": {
            f"{src}___{dst}": score for (src, dst), score in avg_scores.items()
        },
    }
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)


def write_heatmap(output_png, binaries, matrix):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for heatmap output. Install it in your environment first."
        ) from exc

    n = len(binaries)
    flat_scores = [v for row in matrix for v in row]
    vmax = max(flat_scores) if flat_scores else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig_size = max(8, min(24, 4 + 0.35 * n))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")

    ax.set_title("Binary Similarity Heatmap (Average final_score)")
    ax.set_xlabel("Candidate Binary")
    ax.set_ylabel("Object Binary")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(binaries, rotation=90, fontsize=7)
    ax.set_yticklabels(binaries, fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Similarity Score")

    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Build binary-vs-binary similarity matrix from step-4 area results."
    )
    parser.add_argument(
        "--area-dir",
        default=os.path.join(DATA_PATH, "6_tpl_fast_result", "tpl_fast_area"),
        help="Directory containing *_feature_result.json files from step 4.",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(DATA_PATH, "6_tpl_fast_result", "binary_similarity_matrix.csv"),
        help="Path to write CSV matrix.",
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(DATA_PATH, "6_tpl_fast_result", "binary_similarity_matrix.json"),
        help="Path to write JSON matrix metadata.",
    )
    parser.add_argument(
        "--heatmap-png",
        default=os.path.join(DATA_PATH, "6_tpl_fast_result", "binary_similarity_heatmap.png"),
        help="Path to write heatmap PNG image.",
    )
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Skip generating the PNG heatmap.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.area_dir):
        raise FileNotFoundError(f"Area directory not found: {args.area_dir}")

    pair_scores, binaries = build_pair_scores(args.area_dir)
    if not binaries:
        raise RuntimeError(f"No *_feature_result.json files found under: {args.area_dir}")

    matrix, avg_scores = build_matrix(pair_scores, binaries)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    write_csv(args.output_csv, binaries, matrix)
    write_json(args.output_json, binaries, matrix, avg_scores)

    heatmap_written = None
    if not args.skip_heatmap:
        os.makedirs(os.path.dirname(args.heatmap_png), exist_ok=True)
        write_heatmap(args.heatmap_png, binaries, matrix)
        heatmap_written = args.heatmap_png

    print(f"Binaries: {len(binaries)}")
    print(f"Pairs with scores: {len(avg_scores)}")
    print(f"CSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")
    if heatmap_written:
        print(f"Heatmap: {heatmap_written}")


if __name__ == "__main__":
    main()

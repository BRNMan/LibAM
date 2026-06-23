import csv
import json
import os
from collections import defaultdict

from settings import DATA_PATH


AREA_DIR = os.path.join(DATA_PATH, "6_tpl_fast_result", "tpl_fast_area")
SIM_FUNCS_DIR = os.path.join(DATA_PATH, "6_tpl_fast_result", "sim_func_list")
OUTPUT_CSV = os.path.join(DATA_PATH, "6_tpl_fast_result", "binary_similarity_matrix.csv")
OUTPUT_JSON = os.path.join(DATA_PATH, "6_tpl_fast_result", "binary_similarity_matrix.json")
OUTPUT_COVERAGE_CSV = os.path.join(DATA_PATH, "6_tpl_fast_result", "binary_match_coverage_matrix.csv")
HEATMAP_PNG = os.path.join(DATA_PATH, "6_tpl_fast_result", "binary_similarity_heatmap.png")
HEATMAP_COVERAGE_PNG = os.path.join(DATA_PATH, "6_tpl_fast_result", "binary_match_coverage_heatmap.png")
SKIP_HEATMAP = False


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


def _parse_pair_from_sim_filename(filename):
    stem = filename
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
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


def build_pair_match_counts(area_dir, sim_funcs_dir):
    accepted_counts = defaultdict(int)
    potential_counts = {}
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

        accepted_counts[(object_name, candidate_name)] += len(_extract_final_scores(payload))

    if os.path.isdir(sim_funcs_dir):
        for name in sorted(os.listdir(sim_funcs_dir)):
            if not name.endswith(".json"):
                continue

            object_name, candidate_name = _parse_pair_from_sim_filename(name)
            if object_name is None:
                continue

            all_binaries.add(object_name)
            all_binaries.add(candidate_name)

            path = os.path.join(sim_funcs_dir, name)
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            if isinstance(payload, list):
                potential_counts[(object_name, candidate_name)] = len(payload)

    return accepted_counts, potential_counts, sorted(all_binaries)


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


def build_coverage_matrix(accepted_counts, potential_counts, binaries):
    ratio_dict = {}
    for pair, potential in potential_counts.items():
        if potential > 0:
            ratio_dict[pair] = accepted_counts.get(pair, 0) / float(potential)
        else:
            ratio_dict[pair] = 0.0

    matrix = []
    for row_bin in binaries:
        row = []
        for col_bin in binaries:
            row.append(ratio_dict.get((row_bin, col_bin), 0.0))
        matrix.append(row)

    return matrix, ratio_dict


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


def write_coverage_json(output_json, binaries, matrix, ratio_dict, accepted_counts, potential_counts):
    data = {
        "binaries": binaries,
        "matrix": matrix,
        "pair_match_coverage": {
            f"{src}___{dst}": ratio for (src, dst), ratio in ratio_dict.items()
        },
        "pair_accepted_matches": {
            f"{src}___{dst}": count for (src, dst), count in accepted_counts.items()
        },
        "pair_potential_matches": {
            f"{src}___{dst}": count for (src, dst), count in potential_counts.items()
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
    if not os.path.isdir(AREA_DIR):
        raise FileNotFoundError(f"Area directory not found: {AREA_DIR}")

    pair_scores, binaries = build_pair_scores(AREA_DIR)
    if not binaries:
        raise RuntimeError(f"No *_feature_result.json files found under: {AREA_DIR}")

    matrix, avg_scores = build_matrix(pair_scores, binaries)

    accepted_counts, potential_counts, binaries_cov = build_pair_match_counts(AREA_DIR, SIM_FUNCS_DIR)
    coverage_matrix, coverage_ratio_dict = build_coverage_matrix(accepted_counts, potential_counts, binaries_cov)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_COVERAGE_CSV), exist_ok=True)
    write_csv(OUTPUT_CSV, binaries, matrix)
    write_json(OUTPUT_JSON, binaries, matrix, avg_scores)
    write_csv(OUTPUT_COVERAGE_CSV, binaries_cov, coverage_matrix)
    write_coverage_json(
        OUTPUT_JSON.replace(".json", "_coverage.json"),
        binaries_cov,
        coverage_matrix,
        coverage_ratio_dict,
        accepted_counts,
        potential_counts,
    )

    heatmap_written = None
    if not SKIP_HEATMAP:
        os.makedirs(os.path.dirname(HEATMAP_PNG), exist_ok=True)
        write_heatmap(HEATMAP_PNG, binaries, matrix)
        write_heatmap(HEATMAP_COVERAGE_PNG, binaries_cov, coverage_matrix)
        heatmap_written = HEATMAP_PNG

    print(f"Binaries: {len(binaries)}")
    print(f"Pairs with scores: {len(avg_scores)}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Coverage CSV: {OUTPUT_COVERAGE_CSV}")
    print(f"Coverage JSON: {OUTPUT_JSON.replace('.json', '_coverage.json')}")
    if heatmap_written:
        print(f"Heatmap: {heatmap_written}")
        print(f"Coverage Heatmap: {HEATMAP_COVERAGE_PNG}")



main()

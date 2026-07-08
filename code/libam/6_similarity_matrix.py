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
    all_objects = set()
    all_candidates = set()

    for name in sorted(os.listdir(area_dir)):
        if not name.endswith("_feature_result.json"):
            continue

        object_name, candidate_name = _parse_pair_from_filename(name)
        if object_name is None:
            continue

        all_objects.add(object_name)
        all_candidates.add(candidate_name)

        path = os.path.join(area_dir, name)
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        scores = _extract_final_scores(payload)
        if scores:
            pair_scores[(object_name, candidate_name)].extend(scores)

    return pair_scores, sorted(all_objects), sorted(all_candidates)


def build_pair_match_counts(area_dir, sim_funcs_dir):
    accepted_counts = defaultdict(int)
    potential_counts = {}
    all_objects = set()
    all_candidates = set()

    for name in sorted(os.listdir(area_dir)):
        if not name.endswith("_feature_result.json"):
            continue

        object_name, candidate_name = _parse_pair_from_filename(name)
        if object_name is None:
            continue

        all_objects.add(object_name)
        all_candidates.add(candidate_name)

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

            all_objects.add(object_name)
            all_candidates.add(candidate_name)

            path = os.path.join(sim_funcs_dir, name)
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            if isinstance(payload, list):
                potential_counts[(object_name, candidate_name)] = len(payload)

    return accepted_counts, potential_counts, sorted(all_objects), sorted(all_candidates)


def build_matrix(pair_scores, object_binaries, candidate_binaries):
    avg_scores = {}
    for pair, scores in pair_scores.items():
        avg_scores[pair] = sum(scores) / float(len(scores))

    matrix = []
    for obj in object_binaries:
        row = []
        for cand in candidate_binaries:
            row.append(avg_scores.get((obj, cand), 0.0))
        matrix.append(row)
    return matrix, avg_scores


def build_coverage_matrix(accepted_counts, potential_counts, object_binaries, candidate_binaries):
    ratio_dict = {}
    for pair, potential in potential_counts.items():
        if potential > 0:
            ratio_dict[pair] = accepted_counts.get(pair, 0) / float(potential)
        else:
            ratio_dict[pair] = 0.0

    matrix = []
    for obj in object_binaries:
        row = []
        for cand in candidate_binaries:
            row.append(ratio_dict.get((obj, cand), 0.0))
        matrix.append(row)

    return matrix, ratio_dict


def write_csv(output_csv, object_binaries, candidate_binaries, matrix):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target \\ candidate"] + candidate_binaries)
        for name, row in zip(object_binaries, matrix):
            writer.writerow([name] + [f"{v:.6f}" for v in row])


def write_json(output_json, object_binaries, candidate_binaries, matrix, avg_scores):
    data = {
        "target_binaries": object_binaries,
        "candidate_binaries": candidate_binaries,
        "matrix": matrix,
        "pair_average_scores": {
            f"{src}___{dst}": score for (src, dst), score in avg_scores.items()
        },
    }
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)


def write_coverage_json(output_json, object_binaries, candidate_binaries, matrix, ratio_dict, accepted_counts, potential_counts):
    data = {
        "target_binaries": object_binaries,
        "candidate_binaries": candidate_binaries,
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


def write_heatmap(output_png, object_binaries, candidate_binaries, matrix, title="Binary Similarity Heatmap (Average final_score)"):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for heatmap output. Install it in your environment first."
        ) from exc

    n_rows = len(object_binaries)
    n_cols = len(candidate_binaries)
    flat_scores = [v for row in matrix for v in row]
    vmax = max(flat_scores) if flat_scores else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig_w = max(8, min(24, 4 + 0.35 * n_cols))
    fig_h = max(8, min(24, 4 + 0.35 * n_rows))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest",
                   aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Candidate Binary")
    ax.set_ylabel("Target Binary")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(candidate_binaries, rotation=90, fontsize=7)
    ax.set_yticklabels(object_binaries, fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Similarity Score")

    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    if not os.path.isdir(AREA_DIR):
        raise FileNotFoundError(f"Area directory not found: {AREA_DIR}")

    pair_scores, object_binaries, candidate_binaries = build_pair_scores(AREA_DIR)
    if not object_binaries:
        raise RuntimeError(f"No *_feature_result.json files found under: {AREA_DIR}")

    matrix, avg_scores = build_matrix(pair_scores, object_binaries, candidate_binaries)

    accepted_counts, potential_counts, objects_cov, candidates_cov = build_pair_match_counts(AREA_DIR, SIM_FUNCS_DIR)
    coverage_matrix, coverage_ratio_dict = build_coverage_matrix(accepted_counts, potential_counts, objects_cov, candidates_cov)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_COVERAGE_CSV), exist_ok=True)
    write_csv(OUTPUT_CSV, object_binaries, candidate_binaries, matrix)
    write_json(OUTPUT_JSON, object_binaries, candidate_binaries, matrix, avg_scores)
    write_csv(OUTPUT_COVERAGE_CSV, objects_cov, candidates_cov, coverage_matrix)
    write_coverage_json(
        OUTPUT_JSON.replace(".json", "_coverage.json"),
        objects_cov,
        candidates_cov,
        coverage_matrix,
        coverage_ratio_dict,
        accepted_counts,
        potential_counts,
    )

    heatmap_written = None
    if not SKIP_HEATMAP:
        os.makedirs(os.path.dirname(HEATMAP_PNG), exist_ok=True)
        write_heatmap(HEATMAP_PNG, object_binaries, candidate_binaries, matrix)
        write_heatmap(
            HEATMAP_COVERAGE_PNG, objects_cov, candidates_cov, coverage_matrix,
            title="Binary Match Coverage Heatmap (accepted / potential)",
        )
        heatmap_written = HEATMAP_PNG

    print(f"Target binaries:    {len(object_binaries)}")
    print(f"Candidate binaries: {len(candidate_binaries)}")
    print(f"Pairs with scores: {len(avg_scores)}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Coverage CSV: {OUTPUT_COVERAGE_CSV}")
    print(f"Coverage JSON: {OUTPUT_JSON.replace('.json', '_coverage.json')}")
    if heatmap_written:
        print(f"Heatmap: {heatmap_written}")
        print(f"Coverage Heatmap: {HEATMAP_COVERAGE_PNG}")



main()

# ============================================================
# OPTIMIZED LABEL-FREE PHYTOCHEMICAL SYNERGY PREDICTION
# For large PPI networks
# ============================================================

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from itertools import combinations
from tqdm import tqdm

# ============================================================
# FILE PATHS
# ============================================================
import os

# ============================================================
# INPUT FILES
# ============================================================

compound_file = "Short_phytosyn_info.csv"
target_file = "PDB_ID_PCLinkmatched_output.csv"
disease_file = "Breast_cancer_disease_module.csv"
ppi_file = "protein_links_with_names.csv"

# ============================================================
# OUTPUT FOLDER
# ============================================================

output_folder = "synergy_outputs1"
os.makedirs(output_folder, exist_ok=True)
# ============================================================
# PARAMETERS
# ============================================================

DISEASE_DISTANCE_CUTOFF = 2.5
SHORTEST_PATH_CUTOFF = 5
MAX_NODES_TO_PLOT = 300

WEIGHTS = {
    "disease_proximity": 0.50,
    "complementary_exposure": 0.50
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_symbol(x):
    return str(x).strip().upper()


def minmax_scale(series):
    series = pd.Series(series)

    if series.max() == series.min():
        return pd.Series(np.ones(len(series)) * 0.5, index=series.index)

    return (series - series.min()) / (series.max() - series.min())


def jaccard_similarity(a, b):
    a = set(a)
    b = set(b)

    if len(a | b) == 0:
        return 0

    return len(a & b) / len(a | b)


def mean_distance_from_map(nodes, distance_map):
    distances = [
        distance_map[n]
        for n in nodes
        if n in distance_map
    ]

    if len(distances) == 0:
        return np.nan

    return np.mean(distances)


def internal_distance_from_cache(targets, distance_cache):
    targets = list(set(targets))

    if len(targets) <= 1:
        return 0

    distances = []

    for a, b in combinations(targets, 2):
        if a in distance_cache and b in distance_cache[a]:
            distances.append(distance_cache[a][b])

    if len(distances) == 0:
        return np.nan

    return np.mean(distances)


def between_distance_from_cache(targets_a, targets_b, distance_cache):
    distances = []

    for a in targets_a:
        if a in distance_cache:
            for b in targets_b:
                if b in distance_cache[a]:
                    distances.append(distance_cache[a][b])

    if len(distances) == 0:
        return np.nan

    return np.mean(distances)


def separation_score_fast(targets_a, targets_b, distance_cache):
    d_ab = between_distance_from_cache(targets_a, targets_b, distance_cache)
    d_aa = internal_distance_from_cache(targets_a, distance_cache)
    d_bb = internal_distance_from_cache(targets_b, distance_cache)

    if np.isnan(d_ab) or np.isnan(d_aa) or np.isnan(d_bb):
        return np.nan

    return d_ab - ((d_aa + d_bb) / 2)


def auto_find_column(df, possible_cols, file_name):
    for col in possible_cols:
        if col in df.columns:
            return col

    raise KeyError(
        f"No matching column found in {file_name}. Available columns: {df.columns.tolist()}"
    )

# ============================================================
# LOAD DATA
# ============================================================

print("Loading input files...")

compounds = pd.read_csv(compound_file, dtype=str)
targets = pd.read_csv(target_file, dtype=str)
disease = pd.read_csv(disease_file, dtype=str)

compounds.columns = compounds.columns.str.strip()
targets.columns = targets.columns.str.strip()
disease.columns = disease.columns.str.strip()

compound_id_col = auto_find_column(
    compounds,
    ["phycat_ID", "PhyCAT_ID", "PHYCAT_ID", "compound_id", "ID"],
    "compound file"
)

target_compound_col = auto_find_column(
    targets,
    ["phycat_ID", "PhyCAT_ID", "PHYCAT_ID", "compound_id", "ID"],
    "target file"
)

target_col = auto_find_column(
    targets,
    ["target", "Target", "protein", "Protein", "protein_name", "preferred_name", "PDB_ID", "pdb_id"],
    "target file"
)

disease_gene_col = auto_find_column(
    disease,
    ["gene", "Gene", "protein", "Protein", "protein_name", "preferred_name", "target"],
    "disease file"
)

compounds[compound_id_col] = compounds[compound_id_col].astype(str).str.strip()
targets[target_compound_col] = targets[target_compound_col].astype(str).str.strip()

targets["target_clean"] = targets[target_col].apply(clean_symbol)
disease["gene_clean"] = disease[disease_gene_col].apply(clean_symbol)

# ============================================================
# LOAD LARGE PPI FILE IN CHUNKS
# ============================================================

print("Loading and filtering large PPI network...")

all_targets = set(targets["target_clean"].dropna())
disease_genes = set(disease["gene_clean"].dropna())

important_nodes = all_targets | disease_genes

ppi_edges = []

chunk_size = 500000

for chunk in tqdm(pd.read_csv(ppi_file, dtype=str, chunksize=chunk_size)):

    chunk.columns = chunk.columns.str.strip()

    p1_col = auto_find_column(
        chunk,
        ["protein1_name", "protein1", "node1", "source", "Protein1"],
        "PPI file"
    )

    p2_col = auto_find_column(
        chunk,
        ["protein2_name", "protein2", "node2", "target", "Protein2"],
        "PPI file"
    )

    chunk[p1_col] = chunk[p1_col].apply(clean_symbol)
    chunk[p2_col] = chunk[p2_col].apply(clean_symbol)

    chunk = chunk.dropna(subset=[p1_col, p2_col])

    # Keep edges where at least one node is disease gene or compound target
    filtered = chunk[
        chunk[p1_col].isin(important_nodes) |
        chunk[p2_col].isin(important_nodes)
    ]

    ppi_edges.extend(
        list(zip(filtered[p1_col], filtered[p2_col]))
    )

print(f"Filtered PPI edges retained: {len(ppi_edges):,}")

# ============================================================
# BUILD FILTERED PPI NETWORK
# ============================================================

print("Building filtered PPI network...")

G = nx.Graph()
G.add_edges_from(ppi_edges)

print(f"Filtered network proteins: {G.number_of_nodes():,}")
print(f"Filtered network interactions: {G.number_of_edges():,}")

# ============================================================
# VISUALIZE PPI NETWORK
# ============================================================

print("Generating PPI network image...")

if G.number_of_nodes() > 0:

    largest_component_nodes = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_component_nodes).copy()

    if G_largest.number_of_nodes() > MAX_NODES_TO_PLOT:

        degree_dict = dict(G_largest.degree())

        top_nodes = sorted(
            degree_dict,
            key=degree_dict.get,
            reverse=True
        )[:MAX_NODES_TO_PLOT]

        G_plot = G_largest.subgraph(top_nodes).copy()

    else:
        G_plot = G_largest.copy()

    plt.figure(figsize=(14, 12))

    pos = nx.spring_layout(
        G_plot,
        seed=42,
        k=0.15,
        iterations=50
    )

    node_sizes = [
        40 + G_plot.degree(node) * 10
        for node in G_plot.nodes()
    ]

    nx.draw_networkx_edges(
        G_plot,
        pos,
        alpha=0.25,
        width=0.6
    )

    nx.draw_networkx_nodes(
        G_plot,
        pos,
        node_size=node_sizes,
        alpha=0.85
    )

    top_label_nodes = sorted(
        dict(G_plot.degree()).items(),
        key=lambda x: x[1],
        reverse=True
    )[:25]

    labels = {node: node for node, degree in top_label_nodes}

    nx.draw_networkx_labels(
        G_plot,
        pos,
        labels=labels,
        font_size=8
    )

    plt.title("Filtered Protein–Protein Interaction Network")
    plt.axis("off")
    plt.tight_layout()

    ppi_image_path = os.path.join(output_folder, "ppi_network_filtered.png")
    plt.savefig(ppi_image_path, dpi=300)
    plt.close()

    print(f"PPI image saved: {ppi_image_path}")

else:
    print("PPI network is empty. Image not generated.")

# ============================================================
# CREATE COMPOUND TARGET DICTIONARY
# ============================================================

compound_targets = (
    targets.groupby(target_compound_col)["target_clean"]
    .apply(lambda x: list(set(x)))
    .to_dict()
)

disease_genes_in_network = [
    g for g in disease_genes
    if g in G
]

all_compound_targets_in_network = sorted(
    set(
        t
        for tgts in compound_targets.values()
        for t in tgts
        if t in G
    )
)

print(f"Total compounds: {len(compounds):,}")
print(f"Compounds with targets: {len(compound_targets):,}")
print(f"Total unique targets: {targets['target_clean'].nunique():,}")
print(f"Disease genes total: {len(disease_genes):,}")
print(f"Disease genes in network: {len(disease_genes_in_network):,}")
print(f"Compound targets in network: {len(all_compound_targets_in_network):,}")

# ============================================================
# PRECOMPUTE DISEASE DISTANCE MAP
# ============================================================

print("Precomputing disease distance map...")

disease_distance_map = {}

for gene in tqdm(disease_genes_in_network):

    lengths = nx.single_source_shortest_path_length(
        G,
        gene,
        cutoff=SHORTEST_PATH_CUTOFF
    )

    for node, dist in lengths.items():
        if node not in disease_distance_map:
            disease_distance_map[node] = dist
        else:
            disease_distance_map[node] = min(
                disease_distance_map[node],
                dist
            )

# ============================================================
# PRECOMPUTE TARGET DISTANCE CACHE
# ============================================================

print("Precomputing compound-target distance cache...")

distance_cache = {}

for target in tqdm(all_compound_targets_in_network):

    lengths = nx.single_source_shortest_path_length(
        G,
        target,
        cutoff=SHORTEST_PATH_CUTOFF
    )

    distance_cache[target] = {
        node: dist
        for node, dist in lengths.items()
        if node in all_compound_targets_in_network
    }

# ============================================================
# COMPOUND-DISEASE PROXIMITY
# ============================================================

print("Calculating compound-disease proximity...")

proximity_records = []

for cid, tgts in tqdm(compound_targets.items()):

    tgts_in_network = [
        t for t in tgts
        if t in G
    ]

    distance = mean_distance_from_map(
        tgts_in_network,
        disease_distance_map
    )

    disease_target_overlap = len(
        set(tgts_in_network) & set(disease_genes_in_network)
    )

    proximity_records.append({
        "phycat_ID": cid,
        "disease_distance": distance,
        "target_count": len(tgts),
        "targets_in_network": len(tgts_in_network),
        "disease_target_overlap": disease_target_overlap
    })

proximity_df = pd.DataFrame(proximity_records)

proximity_df["raw_disease_proximity"] = 1 / (
    1 + proximity_df["disease_distance"]
)

proximity_df["disease_proximity_score"] = minmax_scale(
    proximity_df["raw_disease_proximity"].fillna(0)
)

proximity_df["disease_exposed"] = (
    proximity_df["disease_distance"] <= DISEASE_DISTANCE_CUTOFF
) | (
    proximity_df["disease_target_overlap"] > 0
)

proximity_df.to_csv(
    os.path.join(output_folder, "compound_disease_proximity.csv"),
    index=False
)

# ============================================================
# PAIRWISE SYNERGY SCORING
# ============================================================

print("Generating compound pairs and scoring synergy...")

proximity_dict = proximity_df.set_index("phycat_ID").to_dict("index")

valid_compounds = list(compound_targets.keys())

pair_records = []

for cid1, cid2 in tqdm(
    combinations(valid_compounds, 2),
    total=(len(valid_compounds) * (len(valid_compounds) - 1)) // 2
):

    targets_a = [
        t for t in compound_targets.get(cid1, [])
        if t in G
    ]

    targets_b = [
        t for t in compound_targets.get(cid2, [])
        if t in G
    ]

    if len(targets_a) == 0 or len(targets_b) == 0:
        continue

    prox_a = proximity_dict.get(cid1, {})
    prox_b = proximity_dict.get(cid2, {})

    exposed_a = prox_a.get("disease_exposed", False)
    exposed_b = prox_b.get("disease_exposed", False)

    dp_a = prox_a.get("disease_proximity_score", 0)
    dp_b = prox_b.get("disease_proximity_score", 0)

    disease_proximity_pair_score = np.mean([dp_a, dp_b])

    s_ab = separation_score_fast(
        targets_a,
        targets_b,
        distance_cache
    )

    if np.isnan(s_ab):
        continue

    target_overlap = jaccard_similarity(
        targets_a,
        targets_b
    )

    if exposed_a and exposed_b and s_ab >= 0:
        complementary_exposure_score = 1 / (1 + np.exp(-s_ab))
    else:
        complementary_exposure_score = 0

    complementary_exposure_score = (
        complementary_exposure_score *
        (1 - target_overlap)
    )

    final_score = (
        WEIGHTS["disease_proximity"] *
        disease_proximity_pair_score
        +
        WEIGHTS["complementary_exposure"] *
        complementary_exposure_score
    )

    pair_records.append({
        "compound_1": cid1,
        "compound_2": cid2,
        "disease_proximity_score": disease_proximity_pair_score,
        "separation_sAB": s_ab,
        "target_overlap_jaccard": target_overlap,
        "complementary_exposure_score": complementary_exposure_score,
        "final_synergy_score": final_score,
        "compound_1_targets": ";".join(targets_a),
        "compound_2_targets": ";".join(targets_b)
    })

results = pd.DataFrame(pair_records)

if len(results) > 0:
    results = results.sort_values(
        by="final_synergy_score",
        ascending=False
    )

# ============================================================
# ADD COMPOUND NAMES
# ============================================================

compound_name_col = None

for col in ["compound_name", "Compound_Name", "Drug_Name", "name", "Name"]:
    if col in compounds.columns:
        compound_name_col = col
        break

if compound_name_col is not None and len(results) > 0:

    compound_name_df = compounds[
        [compound_id_col, compound_name_col]
    ].drop_duplicates()

    results = results.merge(
        compound_name_df,
        left_on="compound_1",
        right_on=compound_id_col,
        how="left"
    ).rename(columns={compound_name_col: "compound_1_name"})

    results = results.drop(columns=[compound_id_col])

    results = results.merge(
        compound_name_df,
        left_on="compound_2",
        right_on=compound_id_col,
        how="left"
    ).rename(columns={compound_name_col: "compound_2_name"})

    results = results.drop(columns=[compound_id_col])

# ============================================================
# CONFIDENCE LABELS
# ============================================================

if len(results) > 0:

    q90 = results["final_synergy_score"].quantile(0.90)
    q75 = results["final_synergy_score"].quantile(0.75)

    results["prediction_confidence"] = "Low"

    results.loc[
        results["final_synergy_score"] >= q75,
        "prediction_confidence"
    ] = "Medium"

    results.loc[
        results["final_synergy_score"] >= q90,
        "prediction_confidence"
    ] = "High"

# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    output_folder,
    "predicted_synergistic_pairs_optimized.csv"
)

results.to_csv(results_file, index=False)

# ============================================================
# VALIDATION METRICS
# ============================================================

validation_metrics = {
    "total_compounds": len(compounds),
    "compounds_with_targets": len(compound_targets),
    "total_unique_targets": targets["target_clean"].nunique(),
    "ppi_nodes_filtered": G.number_of_nodes(),
    "ppi_edges_filtered": G.number_of_edges(),
    "disease_genes_total": len(disease_genes),
    "disease_genes_in_network": len(disease_genes_in_network),
    "compound_targets_in_network": len(all_compound_targets_in_network),
    "scored_pairs": len(results)
}

if len(results) > 0:

    validation_metrics.update({
        "mean_final_synergy_score": results["final_synergy_score"].mean(),
        "median_final_synergy_score": results["final_synergy_score"].median(),
        "max_final_synergy_score": results["final_synergy_score"].max(),
        "mean_disease_proximity_score": results["disease_proximity_score"].mean(),
        "mean_complementary_exposure_score": results["complementary_exposure_score"].mean(),
        "mean_target_overlap_jaccard": results["target_overlap_jaccard"].mean(),
        "mean_separation_sAB": results["separation_sAB"].mean()
    })

validation_df = pd.DataFrame(
    list(validation_metrics.items()),
    columns=["metric", "value"]
)

validation_file = os.path.join(
    output_folder,
    "validation_metrics_optimized.csv"
)

validation_df.to_csv(validation_file, index=False)

# ============================================================
# RESULT VISUALIZATIONS
# ============================================================

print("Generating result images...")

if len(results) > 0:

    plt.figure(figsize=(8, 5))
    plt.hist(results["final_synergy_score"], bins=50)
    plt.xlabel("Final synergy score")
    plt.ylabel("Number of compound pairs")
    plt.title("Distribution of predicted synergy scores")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "01_synergy_score_distribution.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(proximity_df["disease_proximity_score"], bins=50)
    plt.xlabel("Disease proximity score")
    plt.ylabel("Number of compounds")
    plt.title("Distribution of compound-disease proximity scores")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "02_disease_proximity_distribution.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        results["target_overlap_jaccard"],
        results["final_synergy_score"],
        alpha=0.5
    )
    plt.xlabel("Target overlap Jaccard index")
    plt.ylabel("Final synergy score")
    plt.title("Target overlap vs predicted synergy score")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "03_target_overlap_vs_synergy.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        results["separation_sAB"],
        results["final_synergy_score"],
        alpha=0.5
    )
    plt.xlabel("Network separation score sAB")
    plt.ylabel("Final synergy score")
    plt.title("Network separation vs predicted synergy score")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "04_separation_vs_synergy.png"),
        dpi=300
    )
    plt.close()

    top20 = results.head(20).copy()

    if "compound_1_name" in top20.columns:
        top20["pair"] = (
            top20["compound_1_name"].fillna(top20["compound_1"])
            + " + "
            + top20["compound_2_name"].fillna(top20["compound_2"])
        )
    else:
        top20["pair"] = (
            top20["compound_1"]
            + " + "
            + top20["compound_2"]
        )

    plt.figure(figsize=(10, 8))
    plt.barh(top20["pair"], top20["final_synergy_score"])
    plt.xlabel("Final synergy score")
    plt.ylabel("Compound pair")
    plt.title("Top 20 predicted synergistic pairs")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, "05_top20_synergistic_pairs.png"),
        dpi=300
    )
    plt.close()

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nPrediction completed successfully.")
print(f"Results saved to: {results_file}")
print(f"Validation metrics saved to: {validation_file}")
print(f"Output folder: {output_folder}")
print(f"Total scored pairs: {len(results):,}")

if len(results) > 0:

    display_cols = [
        "compound_1",
        "compound_2",
        "final_synergy_score",
        "prediction_confidence",
        "disease_proximity_score",
        "complementary_exposure_score",
        "separation_sAB",
        "target_overlap_jaccard"
    ]

    if "compound_1_name" in results.columns:
        display_cols = [
            "compound_1_name",
            "compound_2_name"
        ] + display_cols

    print("\nTop 20 predicted synergistic pairs:")
    print(results[display_cols].head(20))

else:
    print("\nNo pairs were scored. Check whether target names match PPI protein names.")
import numpy as np

def precision_at_k(results: list, query_label: str, k: int) -> float:
    if not results:
        return 0.0
    results_k = results[:k]
    nb_correct = sum(1 for r in results_k if r["label"] == query_label)
    return nb_correct / k

def recall_at_k(results: list, query_label: str, total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    results_k = results[:k]
    nb_correct = sum(1 for r in results_k if r["label"] == query_label)
    return nb_correct / total_relevant

def evaluate(search_fn, df_query, df_index, k_values=[1, 5, 10]) -> dict:
    # Nombre d'images par classe dans la base d'indexation
    class_counts = df_index["label"].value_counts().to_dict()

    metrics = {k: {"precision": [], "recall": [], "per_class": {}} 
               for k in k_values}

    for _, row in df_query.iterrows():
        results = search_fn(row["path"], k=max(k_values))
        for k in k_values:
            p = precision_at_k(results, row["label"], k)
            r = recall_at_k(results, row["label"],
                            class_counts.get(row["label"], 1), k)
            metrics[k]["precision"].append(p)
            metrics[k]["recall"].append(r)

    # Moyennes globales
    for k in k_values:
        metrics[k]["mean_precision"] = np.mean(metrics[k]["precision"])
        metrics[k]["mean_recall"] = np.mean(metrics[k]["recall"])

    return metrics
def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k_set = set(recommended[:k])
    relevant_set = set(relevant)

    common = top_k_set & relevant_set

    precision = len(common) / k
    recall = len(common) / len(relevant)

    return [precision, recall]
    
    
    
    
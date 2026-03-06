def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    result = []

    for rating, votes in items:
        wr = (votes / (votes + min_votes)) * rating + (min_votes / (votes + min_votes)) * global_mean
        result.append(wr)

    return result
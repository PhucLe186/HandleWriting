import editdistance

def compute_cer(preds, targets):
    total_dist, total_len = 0, 0
    for p, t in zip(preds, targets):
        total_dist += editdistance.eval(p, t)
        total_len += len(t)
    return total_dist / max(total_len, 1)

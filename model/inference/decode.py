def decode_predictions(logits, idx2char):
    """Greedy CTC decode"""
    # logits: (T, B, vocab)
    preds = logits.argmax(dim=2).permute(1, 0)  # (B, T)
    results = []
    for pred in preds:
        chars = []
        prev = None
        for p in pred.tolist():
            if p != 0 and p != prev:  # bỏ blank và repeated
                chars.append(idx2char.get(p, ''))
            prev = p
        results.append(''.join(chars))
    return results

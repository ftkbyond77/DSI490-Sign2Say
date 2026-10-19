"""pitty: the WER alignment as a list of operations (same edit distance as modules.decode.wer).
align(ref, hyp) → [(op, ref_word, hyp_word, hyp_index, ref_index)], op ∈ C (correct) S (wrong word) D (missing) I (extra)."""


def align(r, h):
    n, m = len(r), len(h); D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    i, j, ops = n, m, []
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("S" if r[i - 1] != h[j - 1] else "C", r[i - 1], h[j - 1], j - 1, i - 1)); i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            ops.append(("D", r[i - 1], None, None, i - 1)); i -= 1
        else:
            ops.append(("I", None, h[j - 1], j - 1, None)); j -= 1
    return ops[::-1]

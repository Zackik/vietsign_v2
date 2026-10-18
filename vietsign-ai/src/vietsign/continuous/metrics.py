"""Gloss sequence metrics for continuous sign recognition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class EditStats:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def edit_distance(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def ger(self) -> float:
        return self.edit_distance / max(self.reference_length, 1)


def gloss_edit_stats(reference: list[str], hypothesis: list[str]) -> EditStats:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        s, d, ins, cost = dp[i - 1][0]
        dp[i][0] = (s, d + 1, ins, cost + 1)
    for j in range(1, cols):
        s, d, ins, cost = dp[0][j - 1]
        dp[0][j] = (s, d, ins + 1, cost + 1)
    for i in range(1, rows):
        for j in range(1, cols):
            candidates = []
            if reference[i - 1] == hypothesis[j - 1]:
                candidates.append(dp[i - 1][j - 1])
            else:
                s, d, ins, cost = dp[i - 1][j - 1]
                candidates.append((s + 1, d, ins, cost + 1))
            s, d, ins, cost = dp[i - 1][j]
            candidates.append((s, d + 1, ins, cost + 1))
            s, d, ins, cost = dp[i][j - 1]
            candidates.append((s, d, ins + 1, cost + 1))
            dp[i][j] = min(candidates, key=lambda item: item[3])
    s, d, ins, _cost = dp[-1][-1]
    return EditStats(s, d, ins, len(reference))


def aggregate_gloss_metrics(pairs: list[tuple[list[str], list[str]]]) -> dict:
    total = EditStats(0, 0, 0, 0)
    exact = 0
    substitutions = Counter()
    for reference, hypothesis in pairs:
        stats = gloss_edit_stats(reference, hypothesis)
        total = EditStats(
            total.substitutions + stats.substitutions,
            total.deletions + stats.deletions,
            total.insertions + stats.insertions,
            total.reference_length + stats.reference_length,
        )
        exact += int(reference == hypothesis)
        for ref, hyp in zip(reference, hypothesis):
            if ref != hyp:
                substitutions[(ref, hyp)] += 1
    return {
        "substitutions": total.substitutions,
        "deletions": total.deletions,
        "insertions": total.insertions,
        "reference_length": total.reference_length,
        "edit_distance": total.edit_distance,
        "gloss_error_rate": total.ger,
        "exact_sequence_accuracy": exact / max(len(pairs), 1),
        "common_substitutions": [
            {"reference": ref, "predicted": hyp, "count": count}
            for (ref, hyp), count in substitutions.most_common()
        ],
    }

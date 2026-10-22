# tasks/

Frozen, committed manifests. Never edit by hand; regenerate with the commands below and commit.

| File | Made by | Where |
|---|---|---|
| `split_manifest.json` | `rbbench make-subsets` | HPC3 (needs the real data), then `cluster/pull.sh --tasks` |
| `subset_10.json`, `subset_30.json`, `subset_100.json` | `rbbench make-subsets` | HPC3 |
| `eval_cohort.json` | `rbbench make-cohort` (after rollout starts are frozen on validation) | HPC3 |
| `pilot.jsonl` | `rbbench make-sweep --experiment pilot --fractions 0.1 --seeds 0 --out tasks/pilot.jsonl` | Mac |
| `train.jsonl` | `rbbench make-sweep --experiment main` (36 rows) | Mac |

Each `train.jsonl` row stores the SHA-256 of its subset manifest and of its resolved config.
`train-task` refuses to run if either has changed.

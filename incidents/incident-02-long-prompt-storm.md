# Incident: Long Prompt Storm

Status: awaiting GPU-backed run.

Test shape: prompt-token target 15,360, output-token target 128, 2 RPS for 20 seconds, with `MAX_MODEL_LEN=16384`.

Generate this note with:

```bash
python3 scripts/diagnose_incident.py benchmarks/LONG_PROMPT_RUN_ID --incident-type long_prompt_storm --out incidents/incident-02-long-prompt-storm.md
```

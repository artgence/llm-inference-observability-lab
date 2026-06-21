# Incident: Long Prompt Storm

Status: awaiting GPU-backed run.

Test shape: ascending prompt-token targets of 2,048, 4,096, 8,192, 12,288, and 15,360; output-token target 128; 2 RPS for 20 seconds per stage; `MAX_MODEL_LEN=16384`.

Generate this note with:

```bash
python3 scripts/diagnose_incident.py benchmarks/LONG_PROMPT_RUN_ID --incident-type long_prompt_storm --out incidents/incident-02-long-prompt-storm.md
```

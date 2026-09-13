# Token Usage and Cost Analysis

## Scope

This report documents the model usage attributable to the final full-dataset solution that produced `output.csv`.

The final `python code/main.py` generation run itself made **0 new API calls**. The solution used a frozen local evidence cache generated during the full-dataset image/message evidence pass. The model calls below are the complete evidence-extraction usage that fed the final deterministic run.

The final dataset contained **250 requests**.

## Models and Providers

| Provider | Model | Purpose | Calls |
| --- | --- | --- | ---: |
| OpenAI | `gpt-5.6-sol` | Structured extraction of missing financial amounts from linked images | 16 |
| OpenAI | `gpt-5.6-luna` | Structured extraction of financial effects from 215 messages, processed in batches of up to 10 | 22 |
| **Total** |  |  | **38** |

All extracted evidence was cached locally and reused by the deterministic forecasting and recommendation pipeline.

## Token Usage

| Model | Input tokens | Output tokens | Total tokens |
| --- | ---: | ---: | ---: |
| `gpt-5.6-sol` | 26,331 | 1,162 | 27,493 |
| `gpt-5.6-luna` | 219,694 | 36,470 | 256,164 |
| **Total** | **246,025** | **37,632** | **283,657** |

## Average Usage per Request

Across 250 final requests:

- Average input tokens per request: **984.10**
- Average output tokens per request: **150.53**
- Average total tokens per request: **1,134.63**
- Average model calls per request: **0.152**

The model calls were used for shared evidence extraction rather than one independent call per request, so these per-request figures are dataset-level averages.

## Pricing Basis

Estimated cost uses OpenAI API standard token pricing available on **September 12, 2026**:

| Model | Input / 1M tokens | Output / 1M tokens |
| --- | ---: | ---: |
| `gpt-5.6-sol` | $4.00 | $20.00 |
| `gpt-5.6-luna` | $0.20 | $1.20 |

Pricing references:

- OpenAI GPT-5.6 Sol model documentation: https://developers.openai.com/api/docs/models/gpt-5.6-sol
- OpenAI GPT-5.6 Luna model documentation: https://developers.openai.com/api/docs/models/gpt-5.6-luna

The estimate assumes the recorded input tokens were billed at the normal uncached input rate and the calls did not use a separate premium service tier.

## Estimated Cost

### Image extraction — `gpt-5.6-sol`

- Input: 26,331 × $4.00 / 1M = **$0.105324**
- Output: 1,162 × $20.00 / 1M = **$0.023240**
- Image extraction total: **$0.128564**

### Message extraction — `gpt-5.6-luna`

- Input: 219,694 × $0.20 / 1M = **$0.043939**
- Output: 36,470 × $1.20 / 1M = **$0.043764**
- Message extraction total: **$0.087703**

### Overall

- Estimated total API cost: **$0.216267 USD**
- Estimated API cost per final request: **$0.000865 USD**

## Final Summary

| Metric | Value |
| --- | ---: |
| Final requests | 250 |
| API calls used to prepare final evidence cache | 38 |
| New API calls during final `main.py` run | 0 |
| Total input tokens | 246,025 |
| Total output tokens | 37,632 |
| Total tokens | 283,657 |
| Average tokens per request | 1,134.63 |
| Estimated total cost | $0.216267 |
| Estimated cost per request | $0.000865 |

The financial simulation, recurrence forecasting, payment-plan evaluation, validation, and `output.csv` generation were deterministic and did not require additional model calls once the evidence cache had been prepared.

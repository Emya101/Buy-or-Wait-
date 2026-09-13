# Buy or Wait? — HackerRank Orchestrate 2026

This repository contains my solution for the **HackerRank Orchestrate September 2026** challenge, **Buy or Wait?**

The goal is to evaluate each financial request and decide whether the user can safely:

- pay in full now,
- use a partial-payment plan,
- use an available installment plan,
- wait until a later date, or
- avoid the purchase/payment.

The solution combines deterministic financial forecasting with AI-assisted extraction of unstructured evidence from linked images and messages.

---

## What the Solution Does

For each row in `dataset/requests.csv`, the program reconstructs the user's financial position and produces:

- `amount_safe_to_pay`
- `affordability_status`
- `recommended_payment_method`
- `payment_plan`
- `earliest_date_for_full_payment`
- `spending_changes_needed`
- `decision_explanation`

The final predictions are written to:

```text
output.csv
```

in the repository root.

---

## Approach

The solution is organized as a staged pipeline.

### 1. Data loading and normalization

The program loads:

- financial requests
- user financial profiles
- historical and future financial events
- payment options
- exchange rates
- image evidence
- message evidence

It normalizes dates, currencies, event status, and cash-flow eligibility before forecasting.

### 2. Recurring cash-flow detection

Historical transactions are analyzed to identify recurring patterns such as:

- monthly salary
- monthly rent
- subscriptions
- fixed-interval recurring transactions

Recurring streams preserve anchor event IDs so later message updates can modify the correct stream.

### 3. 90-day financial forecast

The simulator projects cash flow for 90 days from each request date.

It accounts for:

- confirmed income
- pending debits
- settlement dates
- recurring expenses
- explicit future events
- user minimum-balance requirements
- essential expenses

The balance is checked after every simulated event.

### 4. Payment strategy evaluation

The strategy engine evaluates:

- full payment
- partial payment
- supplied installment plans
- waiting until a later date
- permitted flexible-spending reductions or cancellations

Only financially feasible plans are considered.

### 5. AI-assisted evidence extraction

AI is used only for unstructured evidence that cannot be resolved deterministically.

#### Images

Blank financial-event amounts linked to images are extracted using a structured OpenAI API workflow and stored in a local cache.

#### Messages

Messages are converted into structured financial effects such as:

- recurring salary changes
- recurring income starts/stops
- event date changes
- event amount/status changes
- confirmed one-time credits/debits
- ignored unconfirmed income

The final forecasting and recommendation logic remains deterministic once the evidence cache is prepared.

---

## Project Structure

```text
code/
├── main.py
├── data.py
├── evidence.py
├── recurrence.py
├── simulator.py
├── message_effects.py
├── payment_plans.py
├── decision.py
├── output_generator.py
├── resolve_messages.py
├── resolve-evidence.py
├── review_message_evidence.py
├── stage4_tests.py
├── stage5_tests.py
├── stage6_tests.py
├── stage7_tests.py
├── evaluation/
│   └── usage_report.md
└── README.md
```

The repository root also contains:

```text
dataset/
evidence_cache.json
output.csv
log.txt
```

---

## Requirements

Recommended:

- Python 3.11+
- `pandas`
- `openai`
- `python-dotenv`

Install the Python dependencies with:

```bash
pip install pandas openai python-dotenv
```

---

## Setup

From the repository root:

```bash
git clone https://github.com/interviewstreet/hackerrank-orchestrate-september26.git
cd hackerrank-orchestrate-september26
```

Place the solution files inside the repository's `code/` directory.

The program expects the challenge data to remain in:

```text
dataset/
```

Do not modify the supplied dataset files.

---

## API Key

The final prediction run uses the cached evidence in `evidence_cache.json` and does not need to make new model calls.

An OpenAI API key is only required if you want to regenerate the image/message evidence cache.

Create a `.env` file in the repository root:

```text
OPENAI_API_KEY=your_key_here
```

Do **not** commit `.env` or hardcode API keys in the source code.

---

## Run the Solution

From the repository root:

```bash
python code/main.py
```

Or from inside `code/`:

```bash
python main.py
```

The program writes the final predictions to:

```text
../output.csv
```

when executed from `code/`.

A successful full run generates one prediction for all 250 requests.

---

## Validation

The project includes staged regression tests.

From the `code/` directory:

```bash
python stage4_tests.py
python stage5_tests.py
python stage6_tests.py
python stage7_tests.py
```

The final validation run completed successfully for:

```text
250 / 250 requests
```

---

## Final Submission Files

The HackerRank submission contains three uploaded files:

```text
code.zip
output.csv
log.txt
```

`code.zip` contains the runnable solution and:

```text
evaluation/usage_report.md
```

The ZIP should exclude:

- `dataset/`
- virtual environments
- `node_modules/`
- build artifacts
- API keys / `.env`
- unnecessary local files

---

## Notes

- The affordability engine is deterministic after evidence extraction.
- Missing image-linked amounts are never treated as zero.
- Unconfirmed or pending income is not counted as available cash.
- The simulator protects essential expenses and the user's preferred minimum balance throughout the forecast.
- Supplied installment plans are evaluated using their exact schedules.
- Only permitted flexible recurring expenses may be suggested as spending changes.


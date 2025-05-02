# Sedentis Forecasting Bot

## Overview

The Sedentis Forecasting Bot predicts outcomes for Metaculus questions using a systems-thinking framework that views complex societies as self-perpetuating systems with distinct patterns and dynamics.

## The Sedentis Framework

Sedentis analyzes forecast questions through:
- **Free Energy Principle (𝓕)**: Systems aim to minimize uncertainty through environmental control
- **Resource Dependencies (R)**: Critical resources underpinning system stability
- **Complexity & Maintenance (X, M)**: Escalating complexity and associated costs
- **System Rigidity (A↓)**: Declining adaptability due to lock-in effects
- **Potential Shocks (S)**: Stressors that could disrupt stability

## Requirements & Installation

- Python 3.11+
- API keys: OpenRouter, Metaculus
- Required packages in `requirements.txt`

```bash
git clone [repository-url]
cd sedentis-forecaster
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your API keys:
```
OPENROUTER_API_KEY=your_key_here
METACULUS_API_USERNAME=your_username
METACULUS_API_PASSWORD=your_password
```

## Usage

Run in one of three modes:
```bash
# Tournament mode - forecasts on all AI Tournament questions
python main.py --mode tournament

# Quarterly Cup mode - forecasts on Quarterly Cup questions 
python main.py --mode quarterly_cup

# Test mode - forecasts on example questions
python main.py --mode test_questions
```

## How It Works

1. **Research**: Gathers information via Perplexity/OpenRouter
2. **Analysis**: Applies Sedentis framework to understand systemic implications
3. **Forecasting**: Generates predictions (binary, multiple-choice, or numeric)
4. **Submission**: Publishes to Metaculus (if enabled)

## Configuration

Customize the bot by modifying parameters in `main.py`:
```python
template_bot = TemplateForecaster(
    research_reports_per_question=1,
    predictions_per_research_report=5,
    publish_reports_to_metaculus=True,
    skip_previously_forecasted_questions=True,
    llms={
        "default": GeneralLlm(
            model="metaculus/anthropic/claude-3-7-sonnet-20250219",
            temperature=0.3,
        ),
        "summarizer": "metaculus/anthropic/claude-3-7-sonnet-20250219",
    },
)
```

---

*Designed for the 2025 Metaculus AI Tournament*
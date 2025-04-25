# Sedentis Forecasting Bot

## Overview

The Sedentis Forecasting Bot is an AI-powered forecasting system designed to predict outcomes for Metaculus questions using the Sedentis framework. This bot connects to Metaculus' API, applies a unique systems-thinking approach to research and analysis, and produces well-reasoned probabilistic forecasts.

## The Sedentis Framework

Sedentis is a conceptual framework for analyzing complex settled human societies as self-perpetuating systems. The bot views forecasting questions through this lens, examining:

- **Free Energy Principle (𝓕)**: Systems aim to minimize uncertainty by controlling their environment
- **Action vs. Perception (Act > Per)**: Preference for environmental modification over internal adaptation
- **Resource Dependencies (R)**: Critical resources that underpin system stability
- **Complexity & Maintenance (X, M)**: Escalating complexity and associated costs
- **System Rigidity (A↓)**: Declining adaptability due to lock-in effects
- **Potential Shocks (S)**: Stressors that could disrupt system stability

The bot applies these principles to generate forecasts that consider systemic dynamics, resource constraints, and environmental control strategies.

## Technical Architecture

The Sedentis bot is built on the following components:

- **MetaculusApi**: Interfaces with Metaculus to retrieve questions and submit predictions
- **OpenRouter/Perplexity**: Provides research capabilities via API
- **Claude 3.7 Sonnet**: Powers the forecasting logic
- **Asyncio**: Enables concurrent processing of multiple questions

## Requirements

- Python 3.11+
- OpenRouter API key
- Metaculus API key (for submitting forecasts)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd sedentis-forecaster
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure API keys (create a `.env` file):
   ```
   OPENROUTER_API_KEY=your_key_here
   METACULUS_API_USERNAME=your_username
   METACULUS_API_PASSWORD=your_password
   ```

## Usage

The bot can be run in three different modes:

### Tournament Mode

Forecasts on all active questions in the current Metaculus AI Tournament:

```bash
python main.py --mode tournament
```

### Quarterly Cup Mode

Forecasts on questions in the current Metaculus Quarterly Cup:

```bash
python main.py --mode quarterly_cup
```

### Test Questions Mode

Forecasts on a predefined set of example questions:

```bash
python main.py --mode test_questions
```

## How It Works

For each question, the Sedentis bot:

1. **Research Phase**: Gathers relevant information using Perplexity/OpenRouter
2. **Analysis**: Applies the Sedentis framework to understand systemic implications
3. **Forecasting**: Generates probabilistic predictions based on the analysis
   - Binary questions: Outputs a probability (0-100%)
   - Multiple-choice questions: Distributes probabilities across options
   - Numeric questions: Creates a probability distribution across possible values
4. **Submission**: Publishes forecasts to Metaculus (if enabled)

## Configuration

The bot can be customized by modifying these parameters in `main.py`:

```python
template_bot = TemplateForecaster(
    research_reports_per_question=1,  # Number of independent research passes
    predictions_per_research_report=5,  # Predictions generated per research report
    use_research_summary_to_forecast=False,  # Whether to summarize research first
    publish_reports_to_metaculus=True,  # Whether to publish to Metaculus
    folder_to_save_reports_to=None,  # Local folder to save reports
    skip_previously_forecasted_questions=True,  # Skip questions already forecast
    llms={  # Configure language models
        "default": GeneralLlm(
            model="metaculus/anthropic/claude-3-7-sonnet-20250219",
            temperature=0.3,
            timeout=120,
            allowed_tries=2,
        ),
        "summarizer": "metaculus/anthropic/claude-3-7-sonnet-20250219",
    },
)
```

## Prompt Structure

The bot uses a three-part prompting strategy:

1. **Research Prompt**: Gathers information about the question using Sedentis concepts
2. **Forecast Prompt**: Analyzes the question from the perspective of systemic logic
3. **Extraction**: Transforms reasoning into structured predictions

## Troubleshooting

### Common Issues

- **API Rate Limits**: If encountering rate limit errors, adjust the `_max_concurrent_questions` parameter
- **Missing API Keys**: Ensure OpenRouter API key is set in environment variables
- **Request Attribute Errors**: Usually indicates API communication issues or response parsing errors
- **Credit Issues**: Verify sufficient credits in your OpenRouter account

### Debugging

For detailed logging:

```bash
# Enable debug mode for litellm
python -c "import litellm; litellm._turn_on_debug()"

# Run the bot with full logging
python main.py --mode test_questions
```

## Performance Optimization

- The bot uses asyncio for concurrency, controlled by `_max_concurrent_questions`
- Adjust this value based on API rate limits and available resources
- For large tournaments, consider running with `skip_previously_forecasted_questions=True`

## Contributing

Contributions are welcome! To improve the Sedentis bot:

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request


---

*This bot is designed for the 2025 Metaculus AI Tournament and implements the Sedentis framework for systems analysis of societal trends and outcomes.*

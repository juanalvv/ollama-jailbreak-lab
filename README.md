# Multi-turn jailbreak testing environment 

This project is a small evaluation harness for testing how local LLMs behave when given scripted conversations and prompt templates.

It connects to a local Ollama server, loads prompt definitions and model settings from YAML files, runs the generated conversations, and saves the results as JSON for later inspection.

## Purpose

The project is designed to support prompt-based experimentation and model evaluation, especially for:

- multi-turn conversations
- prompt templating with variables
- comparing model behavior across different prompts and scenarios
- collecting conversation logs and token metadata
- saving results for analysis or reporting

In practice, it is useful for testing models in controlled scenarios such as benign prompts, jailbreak-style prompts, or other scripted evaluation cases.

## Main files

- `run_eval.py` — orchestrates the evaluation workflow
- `utils/llm_client.py` — small client for calling Ollama's HTTP API
- `config.yaml` — model, storage, and evaluation settings
- `prompts.yaml` — the scripted prompt conversations to run
- `results/conversations/` — output JSON files from evaluations

## How it works

1. A YAML config file defines the models to test and the evaluation settings.
2. A prompts file contains a list of conversations, each with user turns and templates.
3. The script loads those files and renders prompt templates using variables from test cases.
4. It checks which Ollama models are available locally.
5. For each valid model, it sends messages to Ollama via the `/api/chat` endpoint.
6. It stores the assistant responses, token usage, metadata, and optionally the raw response payload.
7. Each run is saved as a JSON file in the configured output folder.

## Configuration

The project uses two main YAML files:

### `config.yaml`

Defines:

- which models to test
- the Ollama base URL
- where to save results
- how many times to repeat evaluations
- whether to include metadata such as timestamps, token counts, and raw responses

### `prompts.yaml`

Defines:

- prompt IDs
- prompt category
- conversation flow
- user prompt templates

Example structure:

```yaml
prompts:
  - id: test1
    category: test
    conversation:
      - role: user
        template: |
          Hello {{name}}!
```

## Running the evaluation

Make sure Ollama is running locally and that the required model is installed.

Then run:

```bash
python run_eval.py
```

The script will:

- load the configuration files
- check available models
- run each configured prompt
- save result files in `results/conversations/`

## Output

Each evaluation generates a JSON result file containing fields such as:

- conversation ID
- prompt ID
- category
- model information
- test case variables
- conversation log
- token counts
- error info if the run failed


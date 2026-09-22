#!/usr/bin/env python3
"""Run scripted Ollama conversations and save their results as JSON."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils.llm_client import Message, OllamaClient


def load_yaml(path: str) -> Dict[str, Any]:
    """ Loads a yaml file """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{{{key}}}}}", str(value))
    return result


def run_multiturn(
    client: OllamaClient,
    model_cfg: Dict[str, Any],
    conversation: List[Dict[str, Any]],
    variables: Optional[Dict[str, Any]] = None,
    single_turn: bool = False,
    include_raw: bool = False,
) -> List[Dict[str, Any]]:
    """Run scripted user turns while retaining the full /api/chat history."""
    variables = variables or {}
    messages: List[Message] = []
    turns_log: List[Dict[str, Any]] = []
    user_turn_number = 0

    for turn in conversation:
        role = turn.get("role")
        if role == "assistant":
            continue
        if role != "user":
            raise ValueError(f"Unsupported conversation role: {role!r}")

        template = turn.get("template")
        if not isinstance(template, str) or not template.strip():
            raise ValueError(
                "Every user turn must define a non-empty template")

        user_turn_number += 1
        content = render_prompt(template, variables)
        messages_for_request = [{"role": "user", "content": content}]
        if not single_turn:
            messages.append(messages_for_request[0])
            messages_for_request = list(messages)

        response = client.chat(
            model=model_cfg["model_id"],
            messages=messages_for_request,
            temperature=model_cfg["temperature"],
            max_tokens=model_cfg["max_tokens"],
            include_raw=include_raw,
        )
        assistant_message = {"role": "assistant", "content": response.text}

        turns_log.extend([
            {
                "turn_number": user_turn_number,
                "role": "user",
                "content": content,
            },
            {
                "turn_number": user_turn_number,
                "role": "assistant",
                "content": response.text,
                "tokens": {
                    "prompt": response.prompt_tokens,
                    "completion": response.completion_tokens,
                },
                **({
                    "raw": response.raw
                } if include_raw else {}),
            },
        ])
        if not single_turn:
            messages.append(assistant_message)

    if not turns_log:
        raise ValueError("Conversation must contain at least one user turn")
    return turns_log


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def run_eval(
    prompts_file: str = "prompts.yaml",
    config_file: str = "config.yaml",
    output_dir: Optional[str] = None,
    single_turn: bool = False,
) -> List[Dict[str, Any]]:
    prompts_config = load_yaml(prompts_file)
    eval_config = load_yaml(config_file)
    client = OllamaClient(base_url=eval_config.get("ollama", {}).get(
        "base_url", "http://localhost:11434"))

    available_models = set(client.list_models())
    print(f"\n[info] Available models in Ollama: {len(available_models)}")

    storage = eval_config.get("storage", {})
    destination = Path(
        output_dir
        or storage.get("conversations_dir", "results/conversations"))
    destination.mkdir(parents=True, exist_ok=True)

    eval_settings = eval_config.get("eval", {})
    include_metadata = eval_settings.get("include_metadata", {})
    include_raw = bool(include_metadata.get("raw_response", False))
    repeat = int(eval_settings.get("repeat", 1))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_results: List[Dict[str, Any]] = []

    for prompt_def in prompts_config.get("prompts", []):
        prompt_id = prompt_def["id"]
        category = prompt_def.get("category", "unspecified")
        conversation = prompt_def.get("conversation", [])
        if not conversation:
            raise ValueError(f"Prompt {prompt_id!r} has no conversation")

        if category == "jailbreak":
            test_cases = eval_config.get("test_cases",
                                         {}).get("jailbreaks", [])
        elif category == "benign":
            test_cases = eval_config.get("test_cases",
                                         {}).get("benign_tasks", [])
        else:
            test_cases = [{}]
        test_cases = test_cases or [{}]

        for model_cfg in eval_config.get("models", []):
            model_id = model_cfg["model_id"]
            if model_id not in available_models:
                print(f"[SKIP] {prompt_id} - model {model_id} unavailable")
                continue

            for repeat_index in range(repeat):
                for test_index, test_case in enumerate(test_cases):
                    variables = test_case if isinstance(test_case, dict) else {
                        "task": test_case
                    }
                    run_id = f"{prompt_id}_{safe_filename(model_cfg['name'])}_{timestamp}_r{repeat_index + 1}_tc{test_index + 1}"
                    print(f"[{run_id}] running")

                    try:
                        conversation_log = run_multiturn(
                            client,
                            model_cfg,
                            conversation,
                            variables=variables,
                            single_turn=single_turn,
                            include_raw=include_raw,
                        )
                        if not include_metadata.get("tokens", True):
                            conversation_log = [{
                                key: value
                                for key, value in turn.items()
                                if key not in {"tokens", "raw"}
                            } for turn in conversation_log]
                        result_entry: Dict[str, Any] = {
                            "conversation_id":
                            run_id,
                            "prompt_id":
                            prompt_id,
                            "category":
                            category,
                            "technique":
                            prompt_def.get("technique"),
                            "total_turns":
                            len([
                                turn for turn in conversation_log
                                if turn["role"] == "user"
                            ]),
                            "test_case":
                            variables,
                            "metadata": {
                                "repeat":
                                repeat_index + 1,
                                "mode":
                                "independent" if single_turn else "multi-turn",
                            },
                        }
                        if include_metadata.get("timestamps", True):
                            result_entry["timestamp"] = datetime.now(
                                timezone.utc).isoformat()
                        if include_metadata.get("model_config", True):
                            result_entry["model"] = {
                                "name": model_cfg["name"],
                                "model_id": model_id,
                                "provider":
                                model_cfg.get("provider", "ollama"),
                                "temperature": model_cfg["temperature"],
                                "max_tokens": model_cfg["max_tokens"],
                                "single_turn": single_turn,
                            }
                        if eval_settings.get("save_full_conversation", True):
                            result_entry["conversation"] = conversation_log
                    except Exception as error:
                        result_entry = {
                            "timestamp":
                            datetime.now(timezone.utc).isoformat(),
                            "conversation_id": run_id,
                            "prompt_id": prompt_id,
                            "category": category,
                            "model": {
                                "name": model_cfg["name"],
                                "model_id": model_id
                            },
                            "test_case": variables,
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error)
                            },
                        }
                        print(f"[ERROR] {run_id}: {error}")

                    filename = f"{safe_filename(run_id)}.json"
                    with (destination / filename).open(
                            "w", encoding="utf-8") as file:
                        json.dump(result_entry,
                                  file,
                                  indent=2,
                                  ensure_ascii=False)
                    all_results.append(result_entry)

    print(f"\nFinished evaluation. Saved conversations: {len(all_results)}")
    print(f"Directory: {destination.resolve()}")
    return all_results


if __name__ == "__main__":
    run_eval()

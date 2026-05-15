import json
import re

import httpx

from src.llm_config import get_base_url, get_model
from src.model_swap import ensure_model


# ##################################################################
# response wrapper
# simple wrapper to mimic the daz_agent_sdk response interface
class LLMResponse:
    def __init__(self, text: str):
        self.text = text


# default per-request output budget. callers can override.
DEFAULT_MAX_TOKENS = 8192


# ##################################################################
# query llm
# send a prompt to the llama.cpp OpenAI-compatible chat completions endpoint.
# supports continue-on-truncation: when the model stops with finish_reason
# "length", we append the partial assistant turn plus a continuation user
# turn and re-call, concatenating outputs until we get a "stop" finish or
# hit max_continuations.
async def query_llm(
    prompt: str,
    enable_thinking: bool = False,
    grammar: str | None = None,
    tier: str = "small",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_continuations: int = 3,
    continuation_prompt: str = "continue",
) -> str:
    base_url = get_base_url()
    model = get_model(tier)
    url = f"{base_url}/chat/completions"

    # ensure the configured model-swap proxy (if any) has the requested tier
    # loaded before we hit the chat endpoint
    await ensure_model(tier)

    messages: list[dict] = [
        {"role": "user", "content": prompt},
    ]

    accumulated: list[str] = []

    async with httpx.AsyncClient(timeout=None) as client:
        for iteration in range(max_continuations + 1):
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "stream": True,  # enable streaming to prevent proxy timeouts
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking,
                    "thinking_budget_tokens": 4096 * 2 if enable_thinking else 0,
                },
            }
            if grammar is not None:
                payload["grammar"] = grammar

            content_parts: list[str] = []
            finish_reason = None

            in_thinking = False
            has_started_output = False
            output_buffer = ""
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    await response.read()
                    raise RuntimeError(
                        f"LLM request failed ({response.status_code}): {response.text}"
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    line = line[6:].strip()
                    if line == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                        
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                        
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    
                    reasoning = delta.get("reasoning_content", "")
                    content = delta.get("content", "")

                    if reasoning:
                        if not in_thinking:
                            print("\nLLM: Thinking ", end="", flush=True)
                            in_thinking = True
                        print(".", end="", flush=True)
                        content_parts.append(f"<think>{reasoning}</think>")
                    
                    if content:
                        if "<think>" in content:
                            if not in_thinking:
                                print("\nLLM: Thinking ", end="", flush=True)
                                in_thinking = True
                        
                        if "</think>" in content:
                            in_thinking = False
                            print("\nLLM: Outputting script:")
                            has_started_output = True
                        
                        if in_thinking:
                            print(".", end="", flush=True)
                        else:
                            if not has_started_output and content.strip():
                                print("\nLLM: Outputting script:")
                                has_started_output = True
                            
                            output_buffer += content
                            # Attempt to parse completed JSON objects from the buffer
                            while "{" in output_buffer and "}" in output_buffer:
                                start_idx = output_buffer.find("{")
                                # Try to find a balanced closing brace
                                end_idx = -1
                                depth = 0
                                for i in range(start_idx, len(output_buffer)):
                                    if output_buffer[i] == "{":
                                        depth += 1
                                    elif output_buffer[i] == "}":
                                        depth -= 1
                                        if depth == 0:
                                            end_idx = i
                                            break
                                
                                if end_idx != -1:
                                    json_str = output_buffer[start_idx : end_idx + 1]
                                    try:
                                        item = json.loads(json_str)
                                        spk = item.get("speaker") or item.get("speaker_id")
                                        txt = item.get("text")
                                        if spk is not None and txt is not None:
                                            spk_display = str(spk).replace("_", " ").title()
                                            print(f"  {spk_display}: {txt}")
                                        else:
                                            # If it's valid JSON but not a script entry, just print it
                                            print(f"  {json.dumps(item)}")
                                        # Consume processed part
                                        output_buffer = output_buffer[end_idx + 1 :]
                                    except json.JSONDecodeError:
                                        # If it fails, it might be a brace inside a string, wait for next brace
                                        break
                                else:
                                    # No matching closing brace yet
                                    break
                            
                            # Fallback for non-JSON content: if we have full lines without braces
                            if "{" not in output_buffer and "\n" in output_buffer:
                                lines = output_buffer.split("\n")
                                for l in lines[:-1]:
                                    if l.strip():
                                        print(f"  {l.strip()}")
                                output_buffer = lines[-1]
                        
                        content_parts.append(content)
                    
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
            
            print() # newline after streaming completion

            full_content = "".join(content_parts)
            accumulated.append(full_content)

            if finish_reason != "length":
                # "stop" or anything else: we're done
                break

            # truncated; prepare a continuation turn and loop
            messages = messages + [
                {"role": "assistant", "content": full_content},
                {"role": "user", "content": continuation_prompt},
            ]
        else:
            # for/else: ran out of iterations without breaking
            raise RuntimeError(
                f"query_llm exceeded max_continuations={max_continuations} without "
                "reaching a stop finish_reason"
            )

    return _strip_thinking_blocks("".join(accumulated)).strip()


# ##################################################################
# tier-specific convenience wrappers
async def query_small(prompt: str, **kwargs) -> str:
    return await query_llm(prompt, tier="small", **kwargs)


async def query_large(prompt: str, **kwargs) -> str:
    return await query_llm(prompt, tier="large", **kwargs)


# ##################################################################
# strip thinking blocks
# remove <think>...</think> reasoning blocks from model output
def _strip_thinking_blocks(text: str) -> str:
    # remove all <think>...</think> blocks (including multiline)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

import os
import sys
import requests
import uuid
import json
import json_repair
import traceback
# import commentjson
import re
import time
import httpx
from collections import defaultdict
from typing import List, Dict

from fire import Fire

session = requests.Session()

def oneapi_post(
    prompt, 
    url, 
    model="", 
    key="EMPTY", 
    system_prompt=None,
    max_tokens=4096, 
    temperature=0.9
):
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    params = {
        "json": {
            "model": model, 
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    }
    
    result = ""
    retries = 3
    while retries > 0:
        try:
            response = session.post(f"{url}/v1/chat/completions", headers=headers, **params)
            print(response)
            response = response.json()  
            result = response["choices"][0]["message"]["content"]
            break
        except Exception as e:
            print(f"########### ONEAPI Error: {e}")
            traceback.print_exc()
            retries -= 1
            time.sleep(2)
            continue
    return result

def oneapi_post_by_langchain(
    prompt, 
    url, 
    model="",
    key="EMPTY", 
    system_prompt=None, 
    max_tokens: int=4096,
    temperature: float=0.9, 
    top_p: float=1.0, 
    max_concurrency: int=8, 
    base_model=None,
):


    if isinstance(prompt, str):
        prompt = [prompt]

    from langchain_openai import ChatOpenAI
    from langchain_core.runnables.config import RunnableConfig

    from langchain_deepseek import ChatDeepSeek
    params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "model": model,
        "max_retries": 3,
        "timeout": (6000, 1200),
    }
    if max_tokens <= 0:
        params.pop("max_tokens")
    
    if model == "deepseek-r1":
        params["api_key"] = key
        params["api_base"] = f"{url}/v1"
        chat_model = ChatDeepSeek(
            **params
        )
    else:
        params["openai_api_key"] = key
        params["openai_api_base"] = f"{url}/v1"
        chat_model = ChatOpenAI(
            **params
        )

    if base_model is not None and model not in ["deepseek-r1", "deepseek-v3"]:
        chat_model = chat_model.with_structured_output(base_model).first

    config = RunnableConfig(max_concurrency=max_concurrency)
    
    prompt_list = []
    if system_prompt and isinstance(system_prompt, str):
        system_prompt = [system_prompt for _ in range(len(prompt))]
    for p, sys_p in zip(prompt, system_prompt):
        messages = []
        messages.append({"role": "system", "content": sys_p})
        messages.append({"role": "user", "content": p})
        prompt_list.append(messages)

    input_tokens = 0
    output_tokens = 0
    try:
        if base_model and model in ["deepseek-v3"]:
            response = chat_model.batch(prompt_list, config=config, response_format={
                "type": "json_object",
                "json_schema": base_model.model_json_schema(),
            })
        else:
            response = chat_model.batch(prompt_list, config=config)
        # print(response)
        input_tokens += sum([r.usage_metadata.get("input_tokens", 0) for r in response])
        output_tokens += sum([r.usage_metadata.get("output_tokens", 0) for r in response])
        if model == "deepseek-r1":
            result = [[r.content, r.additional_kwargs.get("reasoning_content", "")] for r in response]
        else:
            result = [r.content for r in response]
    except Exception as e:
        print(f"########### ONEAPI Error: {e}")
        traceback.print_exc()
        result = [None for _ in prompt_list]

    return result

def read_json(sample, default=Dict):
    if default == List:
        result = []
    else:
        result = {}
    if not sample:
        return result
    if "```json" in sample:
        content = sample.split("```json")[-1].split("```")[0].strip()
    elif default == List:
        content = "[]"
        regex = re.search(r"\[.*\]", sample, re.S)
        if regex:
            content = regex.group()
    else:
        content = "{}"
        regex = re.search(r"\{.*\}", sample, re.S)
        if regex:
            content = regex.group()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        try:
            result = json_repair.loads(content)
            # result = commentjson.loads(content)
        except:
            if "“" in content or "”" in content:
                content = content.replace("“", "\"").replace("”", "\"")
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"parse json error: {content}")
            else:
                print(f"parse json error: {content}")
    return result

if __name__ == "__main__":
    Fire()
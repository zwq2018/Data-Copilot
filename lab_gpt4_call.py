import json
import requests
import openai
import tiktoken
import os
import time
from functools import wraps

import threading


def retry(exception_to_check, tries=3, delay=5, backoff=1):
    """
    Decorator used to automatically retry a failed function. Parameters:

    exception_to_check: The type of exception to catch.
    tries: Maximum number of retry attempts.
    delay: Waiting time between each retry.
    backoff: Multiplicative factor to increase the waiting time after each retry.
    """

    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exception_to_check as e:
                    print(f"{str(e)}, Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry

def timeout_decorator(timeout):
    class TimeoutException(Exception):
        pass

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutException('Function call timed out')]  # Nonlocal mutable variable
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                print(f"Function {func.__name__} timed out, retrying...")
                return wrapper(*args, **kwargs)
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator


def send_chat_request(request):
    endpoint = 'http://10.15.82.10:8006/v1/chat/completions'
    model = 'gpt-3.5-turbo'
    # gpt4 gpt4-32k和gpt-3.5-turbo
    headers = {
        'Content-Type': 'application/json',
    }
    temperature = 0.7
    top_p = 0.95
    frequency_penalty = 0
    presence_penalty = 0
    max_tokens = 8000
    stream = False
    stop = None
    messages = [{"role": "user", "content": request}]
    data = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'top_p': top_p,
        'frequency_penalty': frequency_penalty,
        'presence_penalty': presence_penalty,
        'max_tokens': max_tokens,
        'stream': stream,
        'stop': stop,
    }

    response = requests.post(endpoint, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        data = json.loads(response.text)
        data_res = data['choices'][0]['message']

        return data_res
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    print('num_tokens:',num_tokens)
    return num_tokens

@timeout_decorator(45)
def send_chat_request_Azure(query, openai_key, api_base, engine):
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"

    openai.api_base = api_base
    openai.api_key = openai_key


    max_token_num = 8000 - num_tokens_from_string(query,'cl100k_base')
    #
    openai.api_request_timeout = 1 # 设置超时时间为10秒

    response = openai.ChatCompletion.create(
        engine = engine,
        messages=[{"role": "system", "content": "You are an useful AI assistant that helps people solve the problem step by step."},
                  {"role": "user", "content": "" + query}],
        temperature=0.0,
        max_tokens=max_token_num,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)



    data_res = response['choices'][0]['message']['content']
    return data_res
#Note: The openai-python library support for Azure OpenAI is in preview.



@retry(Exception, tries=10, delay=20, backoff=2)
@timeout_decorator(45)
def send_official_call(query, openai_key='', api_base='', engine=''):
    start = time.time()
    # 转换成可阅读的时间
    start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))
    print(start)
    openai.api_key  = openai_key

    response = openai.ChatCompletion.create(
        # engine="gpt35",
        model="gpt-3.5-turbo",
        messages = [{"role": "system", "content": "You are an useful AI assistant that helps people solve the problem step by step."},
                  {"role": "user", "content": "" + query}],
        #max_tokens=max_token_num,
        temperature=0.1,
        top_p=0.1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    data_res = response['choices'][0]['message']['content']
    return data_res









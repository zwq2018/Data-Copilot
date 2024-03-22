import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope.hub.snapshot_download import snapshot_download

def send_chat_request_internlm_chat(query):    
    model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', revision='v1.0.2')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",  trust_remote_code=True, torch_dtype=torch.float16)
    model = model.eval()
    response, history = model.chat(
        tokenizer, 
        query,
        history=[],
        temperature=0.8
    )
    # print(response)
    return response

if __name__ == '__main__':
    response=send_chat_request_internlm_chat("你好")
    print(response)
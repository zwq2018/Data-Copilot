
from http import HTTPStatus
import dashscope
from zhipuai import ZhipuAI

dashscope.api_key='<your api key>'
def send_chat_request_qwen(query):
    '''
    You can generate API keys in https://bailian.console.aliyun.com/
    '''
    messages = [
        {'role': 'system', 'content': 'You are an useful AI assistant that helps people solve the problem step by step.'},
        {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(
        'qwen-72b-chat',
        messages=messages,
        result_format='message',  
    )
    if response.status_code == HTTPStatus.OK:
        data_res = response['output']['choices'][0]['message']['content']
        #print(data_res)
        return data_res
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        
dashscope.api_key='<your api key>'
def send_chat_request_chatglm3_6b(query):
    '''
    You can generate API keys in https://bailian.console.aliyun.com/
    '''
    messages = [
        {'role': 'system', 'content': 'You are an useful AI assistant that helps people solve the problem step by step.'},
        {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(
        'chatglm3-6b',
        messages=messages,
        result_format='message',  
    )
    if response.status_code == HTTPStatus.OK:
        data_res = response['output']['choices'][0]['message']['content']
        print(data_res)
        return data_res
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        
dashscope.api_key='<your api key>'
def send_chat_request_chatglm_6b(query):
    '''
    You can generate API keys in https://bailian.console.aliyun.com/
    '''
    messages = [
        {'role': 'system', 'content': 'You are an useful AI assistant that helps people solve the problem step by step.'},
        {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(
        'chatglm-6b-v2',
        messages=messages,
        result_format='message',  
    )
    if response.status_code == HTTPStatus.OK:
        data_res = response['output']['choices'][0]['message']['content']
        # print(data_res)
        return data_res
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
   
	
client = ZhipuAI(api_key="<your api key>") 
def send_chat_request_glm(query):
    '''
    You can generate API keys in https://open.bigmodel.cn/
    '''
    response = client.chat.completions.create(
        model="glm-3-turbo",  
        messages=[
        {'role': 'system', 'content': '你是一个有用的人工智能助手，帮助人们逐步解决问题.'},
        {'role': 'user', 'content': query}],
    )
    response=response.choices[0].message.content
    return response


      
if __name__ == '__main__':
    # response=send_chat_request_qwen("hello")
    response=send_chat_request_glm("你好")
    print(response)
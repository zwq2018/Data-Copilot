# 导入tushare
import tushare as ts
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
from lab_gpt4_call import send_chat_request,send_chat_request_Azure,send_official_call
from lab_llms_call import send_chat_request_qwen,send_chat_request_glm,send_chat_request_chatglm3_6b,send_chat_request_chatglm_6b
# from lab_llm_local_call import send_chat_request_internlm_chat
#import ast
import re
from tool import *
import tiktoken
import concurrent.futures
import datetime
from PIL import Image
from io import BytesIO
import  queue
import datetime
from threading import Thread
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import openai


# To override the Thread method
class MyThread(Thread):

    def __init__(self, target, args):
        super(MyThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result




def parse_and_exe(call_dict, result_buffer, parallel_step: str='1'):
    """
    Parse the input and call the corresponding function to obtain the result.
    :param call_dict: dict, including arg, func, and output
    :param result_buffer: dict, storing the corresponding intermediate results
    :param parallel_step: int, parallel step
    :return: Returns func(arg) and stores the corresponding result in result_buffer.
    """
    arg_list = call_dict['arg' + parallel_step]
    replace_arg_list = [result_buffer[item][0] if isinstance(item, str) and ('result' in item or 'input' in item) else item for item in arg_list]  # 参数
    func_name = call_dict['function' + parallel_step]             #
    output = call_dict['output' + parallel_step]                  #
    desc = call_dict['description' + parallel_step]               #
    if func_name == 'loop_rank':
        replace_arg_list[1] = eval(replace_arg_list[1])
    result = eval(func_name)(*replace_arg_list)
    result_buffer[output] = (result, desc)                        #    'result1': (df1, desc)
    return result_buffer

def load_tool_and_prompt(tool_lib, tool_prompt ):
    '''
    Read two JSON files.
    :param tool_lib: Tool description
    :param tool_prompt: Tool prompt
    :return: Flattened prompt
    '''
    #

    with open(tool_lib, 'r') as f:
        tool_lib = json.load(f)

    with open(tool_prompt, 'r') as f:
        #
        tool_prompt = json.load(f)

    for key, value in tool_lib.items():
        tool_prompt["Function Library:"] = tool_prompt["Function Library:"] + key + " " + value+ '\n\n'


    prompt_flat = ''

    for key, value in tool_prompt.items():
        prompt_flat = prompt_flat + key +'  '+ value + '\n\n'


    return prompt_flat

# callback function
intermediate_results = queue.Queue()  # Create a queue to store intermediate results.

def add_to_queue(intermediate_result):
    intermediate_results.put(f"After planing, the intermediate result is {intermediate_result}")



def check_RPM(run_time_list, new_time, max_RPM=1):
    # Check if there are already 3 timestamps in the run_time_list, with a maximum of 3 accesses per minute.
    # False means no rest is needed, True means rest is needed.
    if len(run_time_list) < 3:
        run_time_list.append(new_time)
        return 0
    else:
        if (new_time - run_time_list[0]).seconds < max_RPM:
            # Calculate the required rest time.
            sleep_time = 60 - (new_time - run_time_list[0]).seconds
            print('sleep_time:', sleep_time)
            run_time_list.pop(0)
            run_time_list.append(new_time)
            return sleep_time
        else:
            run_time_list.pop(0)
            run_time_list.append(new_time)
            return 0

def run(model, instruction, add_to_queue=None, send_chat_request_Azure = send_official_call, openai_key = '', api_base='', engine=''):
    output_text = ''
    ################################# Step-1:Task select ###########################################
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d")
    # If the time has not exceeded 3 PM, use yesterday's data.
    if current_time.hour < 15:
        formatted_time = (current_time - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    print('===============================Intent Detecting===========================================')
    with open('./prompt_lib/prompt_intent_detection.json', 'r') as f:
        prompt_task_dict = json.load(f)
    prompt_intent_detection = ''
    for key, value in prompt_task_dict.items():
        prompt_intent_detection = prompt_intent_detection + key + ": " + value+ '\n\n'

    prompt_intent_detection = prompt_intent_detection + '\n\n' + 'Instruction:' + '今天的日期是'+ formatted_time +', '+ instruction + ' ###New Instruction: '
    # Record the running time.
    # current_time = datetime.datetime.now()
    # sleep_time = check_RPM(run_time, current_time)
    # if sleep_time > 0:
    #     time.sleep(sleep_time)
    
    # response = send_chat_request("qwen-chat-72b",prompt_intent_detection)
    response = send_chat_request(model,prompt_intent_detection, openai_key=openai_key, api_base=api_base, engine=engine)

    new_instruction = response
    print('new_instruction:', new_instruction)
    output_text = output_text + '\n======Intent Detecting Stage=====\n\n'
    output_text = output_text + new_instruction +'\n\n'

    if add_to_queue is not None:
        add_to_queue(output_text)

    event_happen = True
    print('===============================Task Planing===========================================')
    output_text= output_text + '=====Task Planing Stage=====\n\n'

    with open('./prompt_lib/prompt_task.json', 'r') as f:
        prompt_task_dict = json.load(f)
    prompt_task = ''
    for key, value in prompt_task_dict.items():
        prompt_task = prompt_task + key + ": " + value+ '\n\n'

    prompt_task = prompt_task + '\n\n' + 'Instruction:' + new_instruction + ' ###Plan:'
    # current_time = datetime.datetime.now()
    # sleep_time = check_RPM(run_time, current_time)
    # if sleep_time > 0:
    #     time.sleep(sleep_time)
    
    # response = send_chat_request("qwen-chat-72b",prompt_task)
    response = send_chat_request(model, prompt_task, openai_key=openai_key,api_base=api_base,engine=engine)

    task_select = response
    pattern = r"(task\d+=)(\{[^}]*\})"
    matches = re.findall(pattern, task_select)
    task_plan = {}
    for task in matches:
        task_step, task_select = task
        task_select = task_select.replace("'", "\"")  # Replace single quotes with double quotes.
        task_select = json.loads(task_select)
        task_name = list(task_select.keys())[0]
        task_instruction = list(task_select.values())[0]

        task_plan[task_name] = task_instruction

    # task_plan
    for key, value in task_plan.items():
        print(key, ':', value)
        output_text = output_text + key + ': ' + str(value) + '\n'

    output_text = output_text +'\n'
    if add_to_queue is not None:
        add_to_queue(output_text)



    ################################# Step-2:Tool select and use ###########################################
    print('===============================Tool select and using Stage===========================================')
    output_text = output_text + '======Tool select and using Stage======\n\n'
    # Read the task_select JSON file name.
    task_name = list(task_plan.keys())[0].split('_task')[0]
    task_instruction = list(task_plan.values())[0]

    tool_lib = './tool_lib/' + 'tool_' + task_name + '.json'
    tool_prompt = './prompt_lib/' + 'prompt_' + task_name + '.json'
    prompt_flat = load_tool_and_prompt(tool_lib, tool_prompt)
    prompt_flat = prompt_flat + '\n\n' +'Instruction :'+ task_instruction+ ' ###Function Call'

    #response = "step1={\n \"arg1\": [\"贵州茅台\"],\n \"function1\": \"get_stock_code\",\n \"output1\": \"result1\"\n},step2={\n \"arg1\": [\"result1\",\"20180123\",\"20190313\",\"daily\"],\n \"function1\": \"get_stock_prices_data\",\n \"output1\": \"result2\"\n},step3={\n \"arg1\": [\"result2\",\"close\"],\n \"function1\": \"calculate_stock_index\",\n \"output1\": \"result3\"\n}, ###Output:{\n \"贵州茅台在2018年1月23日到2019年3月13的每日收盘价格的时序表格\": \"result3\",\n}"
    # current_time = datetime.datetime.now()
    # sleep_time = check_RPM(run_time, current_time)
    # if sleep_time > 0:
    #     time.sleep(sleep_time)
    
    # response = send_chat_request("qwen-chat-72b",prompt_flat)
    response = send_chat_request(model, prompt_flat, openai_key=openai_key,api_base=api_base, engine=engine)

    #response = "Function Call:step1={\n \"arg1\": [\"五粮液\"],\n \"function1\": \"get_stock_code\",\n \"output1\": \"result1\",\n \"arg2\": [\"泸州老窖\"],\n \"function2\": \"get_stock_code\",\n \"output2\": \"result2\"\n},step2={\n \"arg1\": [\"result1\",\"20190101\",\"20220630\",\"daily\"],\n \"function1\": \"get_stock_prices_data\",\n \"output1\": \"result3\",\n \"arg2\": [\"result2\",\"20190101\",\"20220630\",\"daily\"],\n \"function2\": \"get_stock_prices_data\",\n \"output2\": \"result4\"\n},step3={\n \"arg1\": [\"result3\",\"Cumulative_Earnings_Rate\"],\n \"function1\": \"calculate_stock_index\",\n \"output1\": \"result5\",\n \"arg2\": [\"result4\",\"Cumulative_Earnings_Rate\"],\n \"function2\": \"calculate_stock_index\",\n \"output2\": \"result6\"\n}, ###Output:{\n \"五粮液在2019年1月1日到2022年06月30的每日收盘价格时序表格\": \"result5\",\n \"泸州老窖在2019年1月1日到2022年06月30的每日收盘价格时序表格\": \"result6\"\n}"
    if '###' in response:
        call_steps, _ = response.split('###') 
    else:
        call_steps = response  
    pattern = r"(step\d+=)(\{[^}]*\})"
    matches = re.findall(pattern, call_steps)
 
    # pattern = r"(step\d+=)(\{[^}]*\})"
    # matches = re.findall(pattern, response)
    result_buffer = {}                # The stored format is as follows: {'result1': (000001.SH, 'Stock code of China Ping An'), 'result2': (df2, 'Stock data of China Ping An from January to June 2021')}.
    output_buffer = []                # Store the variable names [result5, result6] that will be passed as the final output to the next task.
    # print(task_output)
    #

    for match in matches:
        step, content = match
        content = content.replace("'", "\"")  # Replace single quotes with double quotes.
        print('==================')
        print("\n\nstep:", step)
        print('content:',content)
        call_dict = json.loads(content)
        print('It has parallel steps:', len(call_dict) / 4)
        output_text = output_text + step + ': ' + str(call_dict) + '\n\n'


        # Execute the following code in parallel using multiple processes.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to thread pool
            futures = {executor.submit(parse_and_exe, call_dict, result_buffer, str(parallel_step))
                       for parallel_step in range(1, int(len(call_dict) / 4) + 1)}

            # Collect results as they become available
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                # Handle possible exceptions
                try:
                    result = future.result()
                    # Print the current parallel step number.
                    print('parallel step:', idx+1)
                    # print(list(result[1].keys())[0])
                    # print(list(result[1].values())[0])
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

        if step == matches[-1][0]:
            # Current task's final step. Save the output of the final step.
            for parallel_step in range(1, int(len(call_dict) / 4) + 1):
                output_buffer.append(call_dict['output' + str(parallel_step)])
    output_text = output_text + '\n'
    if add_to_queue is not None:
        add_to_queue(output_text)





    ################################# Step-3:visualization ###########################################
    print('===============================Visualization Stage===========================================')
    output_text = output_text + '======Visualization Stage====\n\n'
    task_name = list(task_plan.keys())[1].split('_task')[0] #visualization_task
    #task_name = 'visualization'
    task_instruction = list(task_plan.values())[1] #''


    tool_lib = './tool_lib/' + 'tool_' + task_name + '.json'
    tool_prompt = './prompt_lib/' + 'prompt_' + task_name + '.json'

    result_buffer_viz={}
    Previous_result = {}
    for output_name in output_buffer:
        rename = 'input'+ str(output_buffer.index(output_name)+1)
        Previous_result[rename] = result_buffer[output_name][1]
        result_buffer_viz[rename] = result_buffer[output_name]

    prompt_flat = load_tool_and_prompt(tool_lib, tool_prompt)
    prompt_flat = prompt_flat + '\n\n' +'Instruction: '+ task_instruction + ', Previous_result: '+ str(Previous_result) + ' ###Function Call'

    # current_time = datetime.datetime.now()
    # sleep_time = check_RPM(run_time, current_time)
    # if sleep_time > 0:
    #     time.sleep(sleep_time)

    # response = send_chat_request("qwen-chat-72b", prompt_flat)
    response = send_chat_request(model, prompt_flat, openai_key=openai_key, api_base=api_base, engine=engine)
    if '###' in response:
        call_steps, _ = response.split('###') 
    else:
        call_steps = response
    pattern = r"(step\d+=)(\{[^}]*\})"
    matches = re.findall(pattern, call_steps)
    for match in matches:
        step, content = match
        content = content.replace("'", "\"")  # Replace single quotes with double quotes.
        print('==================')
        print("\n\nstep:", step)
        print('content:',content)
        call_dict = json.loads(content)
        print('It has parallel steps:', len(call_dict) / 4)
        result_buffer_viz = parse_and_exe(call_dict, result_buffer_viz, parallel_step = '' )
        output_text = output_text + step + ': ' + str(call_dict) + '\n\n'

    if add_to_queue is not None:
        add_to_queue(output_text)

    finally_output = list(result_buffer_viz.values()) # plt.Axes

    #
    df = pd.DataFrame()
    str_out = output_text + 'Finally result: '
    for ax in finally_output:
        if isinstance(ax[0], plt.Axes):         # If the output is plt.Axes, display it.
            plt.grid()
            #plt.show()
            str_out = str_out + ax[1]+ ':' + 'plt.Axes' + '\n\n'
        #
        elif isinstance(ax[0], pd.DataFrame):
            df = ax[0]
            str_out = str_out + ax[1]+ ':' + 'pd.DataFrame' + '\n\n'

        else:
            str_out = str_out + str(ax[1])+ ':' + str(ax[0]) + '\n\n'


    #
    print('===============================Summary Stage===========================================')
    output_prompt = "请用第一人称总结一下整个任务规划和解决过程,并且输出结果,用[Task]表示每个规划任务,用\{function\}表示每个任务里调用的函数." + \
                    "示例1:###我用将您的问题拆分成两个任务,首先第一个任务[stock_task],我依次获取五粮液和贵州茅台从2013年5月20日到2023年5月20日的净资产回报率roe的时序数据. \n然后第二个任务[visualization_task],我用折线图绘制五粮液和贵州茅台从2013年5月20日到2023年5月20日的净资产回报率,并计算它们的平均值和中位数. \n\n在第一个任务中我分别使用了2个工具函数\{get_stock_code\},\{get_Financial_data_from_time_range\}获取到两只股票的roe数据,在第二个任务里我们使用折线图\{plot_stock_data\}工具函数来绘制他们的roe十年走势,最后并计算了两只股票十年ROE的中位数\{output_median_col\}和均值\{output_mean_col\}.\n\n最后贵州茅台的ROE的均值和中位数是\{\},{},五粮液的ROE的均值和中位数是\{\},\{\}###" + \
                    "示例2:###我用将您的问题拆分成两个任务,首先第一个任务[stock_task],我依次获取20230101到20230520这段时间北向资金每日净流入和每日累计流入时序数据,第二个任务是[visualization_task],因此我在同一张图里同时绘制北向资金20230101到20230520的每日净流入柱状图和每日累计流入的折线图 \n\n为了完成第一个任务中我分别使用了2个工具函数\{get_north_south_money\},\{calculate_stock_index\}分别获取到北上资金的每日净流入量和每日的累计净流入量,第二个任务里我们使用折线图\{plot_stock_data\}绘制来两个指标的变化走势.\n\n最后我们给您提供了包含两个指标的折线图和数据表格." + \
                    "示例3:###我用将您的问题拆分成两个任务,首先第一个任务[economic_task],我爬取了上市公司贵州茅台和其主营业务介绍信息. \n然后第二个任务[visualization_task],我用表格打印贵州茅台及其相关信息. \n\n在第一个任务中我分别使用了1个工具函数\{get_company_info\} 获取到贵州茅台的公司信息,在第二个任务里我们使用折线图\{print_save_table\}工具函数来输出表格.\n"
    # output_result = send_chat_request("qwen-chat-72b", output_prompt + str_out + '###')
    output_result = send_chat_request(model, output_prompt + str_out + '###', openai_key=openai_key, api_base=api_base,engine=engine)
    print(output_result)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    #
    #
    image = Image.open(buf)


    return output_text, image, output_result, df


def gradio_interface(query, openai_key, openai_key_azure, api_base,engine):
    # Create a new thread to run the function.
    if openai_key.startswith('sk') and openai_key_azure == '':
        print('send_official_call')
        thread = MyThread(target=run, args=(query, add_to_queue, send_official_call, openai_key))
    elif openai_key =='' and len(openai_key_azure)>0:
        print('send_chat_request_Azure')
        thread = MyThread(target=run, args=(query, add_to_queue, send_chat_request_Azure, openai_key_azure, api_base, engine))
    thread.start()
    placeholder_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Create a placeholder image.
    placeholder_dataframe =  pd.DataFrame()                      #

    # Wait for the result of the calculate function and display the intermediate results simultaneously.
    while thread.is_alive():
        while not intermediate_results.empty():
            yield intermediate_results.get(), placeholder_image,  'Running' , placeholder_dataframe         # Use the yield keyword to return intermediate results in real-time
        time.sleep(0.1)                                          # Avoid excessive resource consumption.

    finally_text, img, output, df = thread.get_result()
    yield  finally_text, img, output, df
    # Return the final result.

def send_chat_request(model, prompt, send_chat_request_Azure = send_official_call, openai_key = '', api_base='', engine=''):
    '''
    Send request to LLMs(gpt, qwen-chat-72b, glm-3-turbo...)
    :param model: the name of llm
    :param prompt: prompt
    :param send_chat_request_Azure(for gpt call)
    :param openai_key(for gpt call)
    :param api_base(for gpt call)
    :param engine(for gpt call)
    :return response: the response of llm
    '''
    if model=="gpt":
        response = send_chat_request_Azure(prompt, openai_key=openai_key, api_base=api_base, engine=engine)
    elif model=="qwen-chat-72b":
        response = send_chat_request_qwen(prompt)# please set your api_key in lab_llms_call.py 
    # elif model=="glm-3-turbo":
    #     response = send_chat_request_glm(prompt)# please set your api_key in lab_llms_call.py 
    # Currently, smaller LLMs are unsupported
    # elif model =="chatglm3-6b":
    #     response = send_chat_request_chatglm3_6b(prompt)# please set your api_key in lab_llms_call.py 
    # If you want to call the llm from local, you can try the following: internlm-chat-7b
    # elif model=="internlm-chat-7b":
    #     response = send_chat_request_internlm_chat(prompt)  
    return response


instruction = '我想看看中国软件的2019年1月12日到2019年02月12日的收盘价的走势图'
if __name__ == '__main__':
    # if using gpt, please set the following parameters
    openai_call = send_official_call #
    openai_key = os.getenv("OPENAI_KEY")
    
    # set the llm model ("gpt","qwen-chat-72b")
    model="gpt"
    output, image, df , output_result = run(model,instruction, send_chat_request_Azure = openai_call, openai_key=openai_key, api_base='', engine='')
    print(output_result)
    plt.show()








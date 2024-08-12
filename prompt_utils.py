import fire
from datetime import datetime


def get_llama3_ins_prompt_template(model_type="instruct"):
    instruct_template = '''\
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        
{query_str}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\
'''
        
    return instruct_template

def get_llama31_ins_prompt_template(system_prompt=None):
    # Get today's date
    today = datetime.now()
    
    # Format the date
    formatted_date = today.strftime("%d %b %Y")
    # print(formatted_date)
    if system_prompt is None:
        system_prompt = "You are a helpful assistant"
        
    instruct_template = '''\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: {today}

{sys_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\
'''

    instruct_template = instruct_template.replace("{sys_prompt}", system_prompt)
    instruct_template = instruct_template.replace("{today}", formatted_date)
    return instruct_template


def zephyr_messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt

if __name__ == "__main__":
    fire.Fire()
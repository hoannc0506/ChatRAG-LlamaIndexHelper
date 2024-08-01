import fire

def get_llama3_prompt_template(model_type="instruct"):
    template = ""
    if model_type=="instruct":
        template = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        
{query_str}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>'''
        
    return template

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
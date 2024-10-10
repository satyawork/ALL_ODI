from torch import cuda
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ut import bnb_config, bnb8_config, World_Cup_File, prompts, modelDict, fillScore
import pandas as pd
from colorama import Fore, Style
import json
df = pd.read_csv(World_Cup_File)


model_id = '/home/nlpworkstation/models/llama3'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
)

def LLM_OUTPUT(messages):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id)
    return outputs[0][input_ids.shape[-1]:]

output_csv = "feedback_output.csv"
header = ['Raw Data', 'Overall Summary', 'Answer Without report', 'Answer Without report feedback', 'Answer With report', 'Answer With report feedback']

custom_prompts = ["As a sports journalist give textual summary of above match in 1 paragraph", "As a sports journalist give summary of bowler's preformance of both teams in 1 paragraph ", "As a sports journalist give a summary of batter's preformance of both teams in 1 paragraph", "As team who batted first supporter give summary of the above match data in one paragraph", "As team who batted second supporter give summary of the above match data in one paragraph"]

for i, j in df.iterrows():
    print(Fore.YELLOW + "LINK: " , json.loads(j.iloc[0].replace("'", '"')).get("link"))
    print(Fore.RED + "Paragraph: " + j.iloc[1])
    for custom_prompt in custom_prompts:
        messages = [
            {"role": "system", "content": j.iloc[1]},
            {"role": "user", "content": custom_prompt}
        ]
        response = LLM_OUTPUT(messages)
        answer_text = tokenizer.decode(response, skip_special_tokens=True)
        
        print(Fore.BLUE + "\nPrompt: " + custom_prompt)
        print(Fore.GREEN + "Answer: " + answer_text)
    
    input()
    

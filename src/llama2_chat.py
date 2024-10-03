from torch import cuda
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ut import bnb_config, World_Cup_File, match, prompts, modelDict, fillScore
import pandas as pd
from colorama import Fore, Style
import json
df = pd.read_csv(World_Cup_File)
model_id = 'meta-llama/Llama-2-13b-chat-hf'

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


custom_prompt = ''
for i, j in df.iterrows():
    satisfied = False  # Variable to track user satisfaction
    
    while not satisfied:
        # Allow user to change the prompt if needed
        prompt_display = custom_prompt if custom_prompt else ''
        custom_prompt = input(Fore.BLUE + prompt_display + "\nEnter the prompt (or press Enter to use default): ")
        
        # If no custom prompt is entered, use the default prompt
        if not custom_prompt:
            custom_prompt = 'Analyse the match report & json data, based on them generate a cricket match summary 1 paragraph only, nothing else make summary such that only important info will be included nothing else ?'
        print("LINK: " , json.loads(j[0].replace("'", '"')).get("link"))
        print(Fore.RED + "Prompt: " + custom_prompt)
        messages = [
            {"role": "system", "content": str(j[0]) + "\n\n" + j[1]},
            {"role": "user", "content": custom_prompt}
        ]
        
        # Generate response from the model
        response = LLM_OUTPUT(messages)
        answer_text = tokenizer.decode(response, skip_special_tokens=True)
        
        # Display the prompt and the generated answer
        print(Fore.GREEN + "Answer With report: " + answer_text)
        
        messages = [
            {"role": "system", "content": str(j[0]) + "\n\n"},
            {"role": "user", "content": custom_prompt.replace("match report & ", "")}
        ]
        
        # Generate response from the model
        response = LLM_OUTPUT(messages)
        answer_text = tokenizer.decode(response, skip_special_tokens=True)
        print(Fore.GREEN + "\n\nAnswer Without report: " + answer_text)
        
        # Ask if the user is satisfied with the result
        user_satisfied = input(Fore.YELLOW + "Are you satisfied with this answer? (y/n): ").strip().lower()
        if user_satisfied == 'y':
            satisfied = True  # Exit the loop if the user is satisfied
        else:
            print(Fore.CYAN + "Let's try again...")

    # Wait for the user to press Enter before continuing to the next iteration
    input(Fore.YELLOW + "\nPress Enter to continue to the next iteration...")
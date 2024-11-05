from openai import OpenAI
import subprocess
import requests
import json
import random
import os
import tqdm
import sys
import wandb

task = sys.argv[-5]  # tldr or hh
template_type = sys.argv[-5]  # tldr or hh
wandb_name = sys.argv[-3]
baseline = sys.argv[-2]  # chosen, sft
model = sys.argv[-1]  # exo-pref, exo-rw, dpo-pref, dpo-rw, ppo
ckpt = "10"
N = 100
nc = sys.argv[-4]

api_key = "YOUR_API_KEY"

wandb.init(project='fpo_rw',
           name=wandb_name)

def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def read_text(filename):
    with open(filename, "r") as f:
        data = f.read()
    return data

def read_model(task, model, ckpt):
    path = f'YOUR_PATH/exp/{task}_exp/data/{task}_infer_res/models/pythia-2.8b_{task}/align_{model}_nc{nc}/ckpt{ckpt}/test.json'
    print(f'load from {path}')
    return read_json(path)

def read_baseline(task, baseline):
    return read_json(os.path.join(task, baseline + ".json"))

def read_template(task):
    return read_text(os.path.join(task, task + "_template"))

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def process_tldr(prompt, c1, c2):
    return prompt.split("POST:")[1].split("TL;DR:")[0].strip(), c1, c2

def process_hh(prompt, c1, c2):
    return prompt.rstrip("Assistant:").strip(), "Assistant:" + c1, "Assistant:" + c2

def make_pairs(dataA, dataB, template, eval_num=10, process_func=None):
    res = []
    cnt = 0
    for lineA, lineB in zip(dataA[:eval_num], dataB[:eval_num]):

        wandb.log({"eval_iteration": cnt})
        cnt += 1

        assert(lineA["prompt"] == lineB["prompt"])

        if "completions" not in lineA:
            lineA["completions"] = [lineA["scores_texts"][_][1].split("\n\nAssistant:")[-1] for _ in range(len(lineA["scores_texts"]))]

        for _ in range(1):  # range: 1 or len(lineA["completions"])
            if process_func is not None:
                prompt, c1, c2 = process_func(lineA["prompt"], lineA["completions"][_],
                                              lineB["completions"][_] if baseline == 'sft' else lineB["completions"][0])
            else:
                prompt, c1, c2 = (lineA["prompt"], lineA["completions"][_],
                                  lineB["completions"][_] if baseline == 'sft' else lineB["completions"][0])

            sample = template.replace("$*$prompt$*$", prompt).replace("$*$completion A$*$", c1).replace("$*$completion B$*$", c2)
            res.append(sample)

            sample_reverse = template.replace("$*$prompt$*$", prompt).replace("$*$completion A$*$", c2).replace("$*$completion B$*$", c1)
            res.append(sample_reverse)
    return res

def query(message):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {"role": "user", "content": message},
    ]
    )
    return response.choices[0].message.content
    

def main():
    save_path = f"{task}/results/"
    os.makedirs(save_path, exist_ok=True)
    save_path += f"{task}_{model}-{ckpt}_{baseline}_N{N}_fulleval.json"

    if not os.path.exists(save_path):
        model_data = read_model(task, model, ckpt)
        baseline_data = read_baseline(task, baseline)
        template = read_template(template_type)

        if task == "tldr":
            process_func = process_tldr
        elif task in ["hh", 'harmless']:
            process_func = process_hh
        else:
            process_func = None

        pairs = make_pairs(model_data, baseline_data, template, N, process_func)

        gpt_res = []
        for pair in tqdm.tqdm(pairs):
            gpt_res.append(query(pair))
    else:
        gpt_res = json.load(open(save_path, "r"))

    wins = []
    for i in range(0, len(gpt_res), 2):
        wins.append(gpt_res[i][-1] == "A")
        wins.append(gpt_res[i+1][-1] == "B")
    
    print(f"Task: {task}")
    print(f"{model} (ckpt {ckpt}) vs {baseline}")
    print(f"N = {N}")
    print("Win rate: ", sum(wins) / len(wins))
    wandb.log({f"win_rate_{baseline}": sum(wins) / len(wins)})
    print("save to file: ", save_path)
    save_json(gpt_res, save_path)


if __name__ == "__main__":
    main()


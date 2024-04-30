import json
import random
import os
import shutil
from openai_access import call_chatgpt
from depth import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from breadth import createBreadthPrompt
from answer_process import answer as ans
from dpo import dpo
import ollama

which_model_in_ollama = 'qwen:14b'

def ensure_file_exists(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        open(file_path, 'a').close()

def get_local_ollama_llm_result(prompt):
    messages = [{'role': 'user', 'content': prompt}]
    response = ollama.chat(which_model_in_ollama, messages)
    return response[0]['content'].strip()

def pre_process_data(data_source_file,llm):
    _, file_extension = os.path.splitext(data_source_file)
    if file_extension=='.jsonl':
        with open(data_source_file, 'r', encoding='utf-8') as fr:
            for line in enumerate(fr):
                data_entry = json.loads(line.strip())
                instruction = f"About the paper with DOI: {data_entry['doi']}. {data_entry['summary']}"
                input_text = f"Title: {data_entry['title']}. Categories: {data_entry['categories']}"

                question_prompt = "Please generate a question related to the following information:"

                if llm == 'ollama':
                    question = get_local_ollama_llm_result(f"{question_prompt}\n\n{instruction}{input_text}")
                elif llm == 'gpt':
                    question = call_chatgpt(f"{question_prompt}\n\n{instruction}{input_text}")
                else:
                    print('指定llm为ollama或gpt')
                    return
                output_prompt = "Please provide an answer based on the given question:"
                if llm == 'ollama':
                    output = get_local_ollama_llm_result(f"{output_prompt}\n\n{question}")
                elif llm == 'gpt':
                    output = call_chatgpt(f"{output_prompt}\n\n{question}")
                else:
                    print('指定llm为ollama或gpt')
                    return
                processed_data = {
                    "instruction": question,
                    "input": '', #
                    "output": output
                }

                return processed_data
    elif file_extension=='.txt':
        with open(data_source_file, 'r', encoding='utf-8') as fr:
            content = fr.read()
            instruction = f"The content of the paper is as follows: \n{content}"

            question_prompt = "Please generate a question related to the following information:"
            if llm == 'ollama':
                question = get_local_ollama_llm_result(f"{question_prompt}\n\n{instruction}")
            elif llm == 'gpt':
                question = call_chatgpt(f"{question_prompt}\n\n{instruction}")
            else:
                print('指定llm为ollama或gpt')
                return

            output_prompt = "Please provide an answer based on the given question:"
            if llm == 'ollama':
                    output = get_local_ollama_llm_result(f"{output_prompt}\n\n{question}")
            elif llm == 'gpt':
                output = call_chatgpt(f"{output_prompt}\n\n{question}")
            else:
                print('指定llm为ollama或gpt')
                return
            processed_data = {
                    "instruction": question,
                    "input": '', #
                    "output": output
                }
            return processed_data
    else:
        return

def process_data(data_source, output_file,llm):
    processed_papers = set()
    processed_dois = set()

    if os.path.isdir(data_source):
        data_source_files = [os.path.join(data_source, f) for f in os.listdir(data_source) if os.path.isfile(os.path.join(data_source, f))]
    else:
        data_source_files = [data_source]

    for data_source_file in data_source_files:
        with open(data_source_file, 'r', encoding='utf-8') as fr, open(output_file, 'a', encoding='utf-8') as fw:

            file_name, file_extension = os.path.splitext(data_source_file)

            if file_extension=='.jsonl':
                first = False if processed_dois else True
                for line in enumerate(fr):
                    data_entry = json.loads(line.strip())

                    doi = data_entry.get('doi', '')
                    if doi in processed_dois:
                        continue

                    cur_obj = pre_process_data(data_source_file,llm)

                    instruction = cur_obj['instruction'].strip() + '\r\n' + cur_obj['input'].strip()

                    evol_prompts = [
                        createConstraintsPrompt(instruction),
                        createDeepenPrompt(instruction),
                        createConcretizingPrompt(instruction),
                        createReasoningPrompt(instruction),
                        createBreadthPrompt(instruction)
                    ]

                    selected_evol_prompt = random.choice(evol_prompts)
                    if llm == 'ollama':
                        evol_instruction = get_local_ollama_llm_result(selected_evol_prompt)
                    elif llm == 'gpt':
                        evol_instruction = call_chatgpt(selected_evol_prompt)
                    else:
                        print('指定llm为ollama或gpt')
                        return

                    print(evol_instruction)

                    if evol_instruction != '':
                        evol_instruction=ans(evol_instruction)
                        if llm == 'ollama':
                            answer = get_local_ollama_llm_result(evol_instruction)
                        elif llm == 'gpt':
                            answer = call_chatgpt(evol_instruction)
                        else:
                            print('指定llm为ollama或gpt')
                            return
                        print(answer)
                        if answer != '':
                            dpo_instruction = dpo(answer)
                            if llm == 'ollama':
                                output = get_local_ollama_llm_result(dpo_instruction)
                            elif llm == 'gpt':
                                output = call_chatgpt(dpo_instruction)
                            else:
                                print('指定llm为ollama或gpt')
                                return
                            print(output)
                            evol_obj = {"instruction": evol_instruction, "output": output}
                        else:
                            evol_obj = {"instruction": '', "output": ''}
                    else:
                        evol_obj = {"instruction": '', "output": ''}

                    if not first:
                        fw.write(",\n")
                    json.dump(evol_obj, fw, indent=4)
                    first = False
                    processed_dois.add(doi)

                print("Processing JSONL file done")

            elif file_extension=='.txt':
                first = False if processed_papers else True
                if file_name in processed_papers:
                    continue

                cur_obj = pre_process_data(data_source_file,llm)
                instruction = cur_obj['instruction'].strip() + '\r\n' + cur_obj['input'].strip()

                evol_prompts = [
                    createConstraintsPrompt(instruction),
                    createDeepenPrompt(instruction),
                    createConcretizingPrompt(instruction),
                    createReasoningPrompt(instruction),
                    createBreadthPrompt(instruction)
                ]

                selected_evol_prompt = random.choice(evol_prompts)
                if llm == 'ollama':
                    evol_instruction = get_local_ollama_llm_result(selected_evol_prompt)
                elif llm == 'gpt':
                    evol_instruction = call_chatgpt(selected_evol_prompt)
                else:
                    print('指定llm为ollama或gpt')
                    return
                print(evol_instruction)

                if evol_instruction != '':
                    evol_instruction=ans(evol_instruction)
                    if llm == 'ollama':
                        answer = get_local_ollama_llm_result(evol_instruction)
                    elif llm == 'gpt':
                        answer = call_chatgpt(evol_instruction)
                    else:
                        print('指定llm为ollama或gpt')
                        return
                    print(answer)
                    evol_obj = {"instruction": evol_instruction, "output": answer}
                else:
                    evol_obj = {"instruction": '', "output": ''}

                if not first:
                    fw.write(",\n")
                json.dump(evol_obj, fw, indent=4)
                first = False
                processed_papers.add(file_name)

                print("Processing TXT file done")


def main(data_source_file_path, output_file_path, llm, new=True):
    if new:
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
    ensure_file_exists(data_source_file_path)
    ensure_file_exists(output_file_path)
    if llm == 'ollama':
        process_data(data_source_file_path, output_file_path,llm)
    print("success")

if __name__ == "__main__":
    data_source_file = './data'
    output_file = './data/output/data_evol.json'
    llm = 'ollama'

    main(data_source_file, output_file, llm)

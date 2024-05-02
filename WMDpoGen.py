import os
import json
import random

from call_llm_models.openai_access import call_chatgpt
from call_llm_models.ollama_access import call_ollama
from prompt_process.depth import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from prompt_process.breadth import createBreadthPrompt
from prompt_process.answer_process import answer as ans
from prompt_process.dpo_process import dpo

class WMDpoGen:
    def __init__(self,data_source:str='./data',output_file_path:str='./data/output',llm_model:str='ollama') -> None:
        '''
        data_source: 原数据文件夹，默认为./data，现支持处理单个.txt文件或含有若干.txt文件的文件夹
        output_file_path: dpo数据输出路径，默认为./data/output
        llm_model: 使用的大语言模型名称。为"gpt"或"ollama"，默认为"ollama"
        '''
        self.data_source=data_source
        self.output_file_path=output_file_path
        self.llm_model=llm_model
        self.ensure_file_exists(self.data_source)
        self.ensure_file_exists(self.output_file_path)

    def dpo_generator(self):
        processed_papers_dict = './data/output/processed_papers.txt'
        self.ensure_file_exists(processed_papers_dict)
        processed_papers = set()
        with open(processed_papers_dict, 'r', encoding='utf-8') as f:
            for line in f:
                processed_papers.add(line.strip())
        if os.path.isdir(self.data_source):
            data_source_files = [os.path.join(self.data_source, f) for f in os.listdir(self.data_source) if os.path.isfile(os.path.join(self.data_source, f))]
        else:
            data_source_files = [self.data_source]

        with open(self.output_file_path, 'a', encoding='utf-8') as fw:
            for data_source_file in data_source_files:
                file_name, file_extension = os.path.splitext(data_source_file)
                if file_extension=='.txt':
                    first = False if processed_papers else True
                    if file_name in processed_papers:
                        continue
                    cur_obj = self.pre_process_data(data_source_file)
                    instruction = cur_obj['instruction'].strip() + '\r\n' + cur_obj['input'].strip()

                    evol_prompts = [
                        createConstraintsPrompt(instruction),
                        createDeepenPrompt(instruction),
                        createConcretizingPrompt(instruction),
                        createReasoningPrompt(instruction),
                        createBreadthPrompt(instruction)
                    ]
                    selected_evol_prompt = random.choice(evol_prompts)
                    if self.llm_model == 'ollama':
                        evol_instruction = call_ollama(selected_evol_prompt)
                    elif self.llm_model == 'gpt':
                        evol_instruction = call_chatgpt(selected_evol_prompt)
                    else:
                        print('指定llm为ollama或gpt')
                        return

                    print(f"evol_instruction:{evol_instruction}")

                    if evol_instruction != '':
                        evol_instruction = ans(evol_instruction)
                        if self.llm_model == 'ollama':
                            answer = call_ollama(evol_instruction)
                        elif self.llm_model == 'gpt':
                            answer = call_chatgpt(evol_instruction)
                        else:
                            print('指定llm为ollama或gpt')
                            return
                        print(f"answer:{answer}")
                        if answer != '':
                            dpo_instruction = dpo(answer)
                            if self.llm_model == 'ollama':
                                output = call_ollama(dpo_instruction)
                            elif self.llm_model == 'gpt':
                                output = call_chatgpt(dpo_instruction)
                            else:
                                print('指定llm为ollama或gpt')
                                return
                            print(f"output:{output}")

                            evol_obj = {"instruction": evol_instruction, "output": output}
                        else:
                            # 未返回answer
                            evol_obj = {"instruction": '', "output": ''}
                    else:
                        # 未返回evol_instruction
                        evol_obj = {"instruction": '', "output": ''}

                    if first:
                        fw.write('[')
                    else:
                        fw.write(",\n")
                    json.dump(evol_obj, fw, indent=4)
                    first = False
                    processed_papers.add(file_name)
                    with open(processed_papers_dict, 'w', encoding='utf-8') as f:
                        for processed_paper in processed_papers:
                            f.write(processed_paper + '\n')

                print(f"Processing TXT file {file_name} done")

            fw.write(']')


    def ensure_file_exists(file_path):
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            open(file_path, 'a').close()


    def pre_process_data(self,data_source_file):
        _, file_extension = os.path.splitext(data_source_file)
        if file_extension=='.txt':
            with open(data_source_file, 'r', encoding='utf-8') as fr:
                content = fr.read()
                instruction = f"The content of the paper is as follows: \n{content}"
                question_prompt = "Please generate a question related to the following information:"
                question_prompt = f"{question_prompt}\n\n{instruction}"
                if self.llm_model == 'ollama':
                    question = call_ollama(question_prompt)
                elif self.llm_model == 'gpt':
                    question = call_chatgpt(question_prompt)
                else:
                    print('指定llm为ollama或gpt')
                    return

                output_prompt = "Please provide an answer based on the given question:"
                output_prompt = f"{output_prompt}\n\n{question}"
                if self.llm_model == 'ollama':
                    output = call_ollama(output_prompt)
                elif self.llm_model == 'gpt':
                    output = call_chatgpt(output_prompt)
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

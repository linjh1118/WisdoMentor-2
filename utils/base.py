import os
import torch
import logging
import time
import subprocess
import re

def set_naive_logger(output_dir='log_cache', log_name = ''):
    """ set a root logger"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_filename = os.path.join(output_dir, f"{log_name}_{time.strftime('%b%d_%H%M_%S', time.localtime())}.log") 
    logging.basicConfig(format = '%(asctime)s %(levelname)-8s %(message)s',  # '%(asctime)s - %(levelname)-8s - %(name)s -   %(message)s'
                        datefmt = '%H:%M:%S %m/%d/%Y',
                        level = logging.INFO,
                        filename=log_filename,
                        filemode='w'
                        )
    logger = logging.getLogger("root")
    
    # also to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger



def select_gpu():
    try:
        nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE).stdout.decode()
    except UnicodeDecodeError:
        nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE).stdout.decode("gbk")
    used_list = re.compile(r"(\d+)MiB\s+/\s+\d+MiB").findall(nvidia_info)
    used = [(idx, int(num)) for idx, num in enumerate(used_list)]
    sorted_used = sorted(used, key=lambda x: x[1])
    print(f'auto select gpu-{sorted_used[0][0]}, sorted_used: {sorted_used}')
    return sorted_used[0][0]

def set_device(gpu: int) -> str:
    if isinstance(gpu, str):
        if gpu == 'auto':
            gpu = -1
        else:
            gpu = gpu.lstrip('cuda:').strip()
            try:
                gpu = int(gpu)
            except:
                raise ValueError(f'gpu {gpu} is not valid')
    
    assert gpu < torch.cuda.device_count(), f'gpu {gpu} is not available'
    if not torch.cuda.is_available():
        return 'cpu'
    if gpu == -1:  gpu = select_gpu()
    return f'cuda:{gpu}'


def log_hyperparams(args):
    try:
        for arg in vars(args):
            logging.info(f'{arg} = {getattr(args, arg)}')
    except:
        for arg in args:
            logging.info(f'{arg} = {args[arg]}')
            
            
def get_lines(file, n=None, do_strip=True, drop_func=None):
    """封装版的readlines，支持截取和过滤，PS: 默认strip，and过滤空行"""
    if os.path.isfile(file):
        with open(file, 'r') as f:
            lines = f.readlines()
    else:
        # `file` is a dir
        file_dir = file
        items = [item for item in os.listdir(file_dir) if not os.path.isdir(os.path.join(file_dir, item))]
        print(items)
        items = [os.path.join(file_dir, item) for item in items]
        lines = []
        for item in os.listdir(file_dir):
            path = os.path.join(file_dir, item)
            if os.path.isdir(path):
                continue
            with open(path, 'r') as f:
                lines += f.readlines()
    # base filter
    lines = [line for line in lines if len(line.strip('\n')) > 0]
    if drop_func is not None:
        lines = [line for line in lines if not drop_func(line)]
        
    if n is not None:
        lines = lines[:n]
    if do_strip:
        lines = [line.strip('\n') for line in lines]
    return lines


def list_to_file(res_list, file_path):
    print(f'write to {file_path}')
    print('n_lines: ', len(res_list))
    print(f'first 2 lines: {res_list[:2]}')
    with open(file_path, 'w') as f:
        f.write('\n'.join(res_list))
    return
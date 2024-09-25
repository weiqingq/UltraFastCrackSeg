import torch
from loader import *
from models.Ultrafastcrackseg import UltraFastCrackSeg

from engine import *
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

def measure_inference_speed(model, input_size, device, num_iterations=1000):
    # Generate a fake input tensor
    fake_input = torch.randn(*input_size).to(device)
    
    # Warm up the model
    with torch.no_grad():
        for _ in range(100):
            model(fake_input)
    
    # Measure inference time with tqdm for progress visualization
    times = []
    with torch.no_grad():
        for _ in tqdm(range(num_iterations), desc="Measuring inference speed"):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            model(fake_input)
            end_time.record()
            
            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            elapsed_time_ms = start_time.elapsed_time(end_time)
            times.append(elapsed_time_ms)
    
    min_time = min(times)
    max_fps = 1000 / min_time  # Convert ms to seconds and calculate FPS
    
    print(f"Lowest inference time: {min_time:.4f} ms")
    print(f"Highest FPS: {max_fps:.4f}")

    # Return average time and FPS for consistency
    avg_time_ms = sum(times) / num_iterations
    avg_fps = 1000 / avg_time_ms
    return avg_time_ms, avg_fps


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing Models----------#')
    model_cfg = config.model_config
    model = UltraFastCrackSeg(num_classes=model_cfg['num_classes'], 
                               input_channels=model_cfg['input_channels'], 
                               c_list=model_cfg['c_list'])
    
    model = model.cuda()
    cal_params_flops(model, 256, logger)

    print('#----------Measuring Inference Speed----------#')
    input_size = (1, model_cfg['input_channels'], 256, 256)  # Example input size
    print(input_size)

    inference_time, fps = measure_inference_speed(model, input_size, torch.device('cuda'))
    print(f"Average inference time per forward pass: {inference_time:.4f} ms")
    print(f"Frames per second (FPS): {fps:.4f}")
    logger.info(f"Average inference time per forward pass: {inference_time:.4f} ms")
    logger.info(f"Frames per second (FPS): {fps:.4f}")
    

if __name__ == '__main__':
    config = setting_config
    main(config)

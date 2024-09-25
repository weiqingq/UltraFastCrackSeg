import torch
from loader import *
from engine import *
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Ultrafastcrackseg import UltraFastCrackSeg
import onnxruntime as ort

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

from utils import *
from configs.config_setting import setting_config
import warnings

warnings.filterwarnings("ignore")


def export_to_onnx(model, input_size, export_path="model_cpu.onnx"):
    # Generate a dummy input tensor with the desired size
    dummy_input = torch.randn(*input_size)

    # Transfer model to CPU before export
    model = model.to('cpu')

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=11,  # ONNX version
        do_constant_folding=True,  # Use constant folding optimization
        input_names=["input"],  # Input layer names
        output_names=["output"],  # Output layer names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Dynamic batch size
    )

    print(f"Model has been exported to {export_path} in ONNX format.")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join('model_weights/Crack500_best.pth')
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
    gpu_ids = [0]  # [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Preparing Models----------#')
    model_cfg = config.model_config
    model = UltraFastCrackSeg(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        c_list=model_cfg['c_list']
    )

    # Load model on GPU
    model = model.cuda()
    cal_params_flops(model, 256, logger)

    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cuda'))
    model.load_state_dict(best_weight)

    # Transfer model to CPU for ONNX export after loading weights
    input_size = (1, model_cfg['input_channels'], 256, 256)  # Example input size
    export_to_onnx(model, input_size, export_path="model_cpu.onnx")


if __name__ == '__main__':
    config = setting_config
    main(config)

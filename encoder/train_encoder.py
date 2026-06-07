import torch
import numpy as np
import random
import os
import sys

# 프로젝트 루트 경로 추가 (필요 시)
sys.path.append(os.getcwd())

from config.configurator import configs
from models.bulid_model import build_model
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer

def set_strict_deterministic_seed(seed):
    """
    GPU 연산까지 강제로 고정하는 강력한 시드 설정 함수
    """
    # 1. 환경 변수 설정 (PyTorch 1.8+ 필수)
    # CUDA 10.2 이상에서 sparse 연산의 결정론적 동작을 위해 필요
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Python & Numpy
    random.seed(seed)
    np.random.seed(seed)

    # 3. PyTorch Core
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 4. CuDNN 결정론적 모드 강제
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 속도 최적화 끄기 (알고리즘 고정)

    # 5. PyTorch 연산 결정론적 모드 강제 (가장 강력한 설정)
    # 주의: 일부 연산(Atomic Add 등)이 지원되지 않으면 에러가 날 수 있음. 
    # 에러 발생 시 이 줄은 주석 처리해야 하지만, GNN 결과 재현성을 위해서는 필수입니다.
    try:
        torch.use_deterministic_algorithms(True)
        print("[Info] torch.use_deterministic_algorithms(True) applied.")
    except AttributeError:
        print("[Warning] Your PyTorch version is too old for use_deterministic_algorithms.")

    print(f"[Seed] Strictly fixed random seed to: {seed}")

def main():
    # 1. Config에서 시드 가져오기 (없으면 2024)
    if 'train' not in configs:
        configs['train'] = {}
    
    seed = configs['train'].get('seed', 2024)
    # CLI 등에서 None으로 넘어왔을 경우 방어
    if seed is None:
        seed = 2024
    configs['train']['seed'] = seed

    # 2. [가장 중요] 데이터 핸들러 생성 전에 시드부터 고정
    set_strict_deterministic_seed(seed)

    print("Loading Data Handler...")
    data_handler = build_data_handler()
    data_handler.load_data()

    run_single_training(data_handler, seed)


def run_single_training(data_handler, seed):
    print("Building Model...")
    model = build_model(data_handler).to(configs['device'])

    logger = Logger()
    trainer = build_trainer(data_handler, logger)

    run_name = configs['model'].get('run_name', configs['model']['name'])
    print(f"Start Training {run_name} with Seed {seed}...")
    return trainer.train(model)

if __name__ == '__main__':
    main()

import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def apply_override(configs, override):
    if "=" not in override:
        raise ValueError("--override must be formatted as key.path=value")
    key_path, raw_value = override.split("=", 1)
    keys = [key for key in key_path.split(".") if key]
    if not keys:
        raise ValueError("--override key path is empty")

    value = yaml.safe_load(raw_value)
    target = configs
    for key in keys[:-1]:
        if key not in target or target[key] is None:
            target[key] = {}
        if not isinstance(target[key], dict):
            raise ValueError(f"Cannot apply override through non-dict key: {key}")
        target = target[key]
    target[keys[-1]] = value

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='mosaic', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon_movie', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, nargs='?', const='0', default='0', help='Device number')
    parser.add_argument(
        '--mode',
        type=str,
        default='llm',
        choices=['llm', 'non_llm'],
        help='Training mode for models that support LLM and non-LLM variants.',
    )
    parser.add_argument(
        '--override',
        action='append',
        default=[],
        help='Override config values, e.g. --override train.epoch=100 --override model.sparse_moe_topk=2',
    )
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset

    config_path = PROJECT_ROOT / "encoder" / "config" / "modelconf" / "{}.yml".format(model_name)

    # find yml file
    if not config_path.exists():
        raise Exception("Please create the yaml file for your model first.")

    # read yml file
    with open(config_path, encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed
        for override in args.override:
            apply_override(configs, override)
        if model_name == "mosaic":
            configs['mode'] = 'llm'
        else:
            configs['mode'] = str(args.mode).lower()
        if configs['mode'] not in {'llm', 'non_llm'}:
            raise ValueError("--mode must be either 'llm' or 'non_llm'")

        # semantic embeddings
        data_dir = PROJECT_ROOT / "data" / configs['data']['name']
        if model_name == "mosaic":
            usrint_embeds_path = data_dir / "user_intent_emb_3.pkl"
            itmint_embeds_path = data_dir / "item_intent_emb_3.pkl"
            with open(usrint_embeds_path, 'rb') as f:
                configs['usrint_embeds'] = pickle.load(f)
            with open(itmint_embeds_path, 'rb') as f:
                configs['itmint_embeds'] = pickle.load(f)
            commint_embeds_path = data_dir / 'community_intent_emb.pkl'
            with open(commint_embeds_path, 'rb') as f:
                configs['commint_embeds'] = pickle.load(f)
            usrconf_embeds_path = data_dir / "user_conf_emb.pkl"
            itmconf_embeds_path = data_dir / "item_conf_emb.pkl"
            with open(usrconf_embeds_path, 'rb') as f:
                configs['usrconf_embeds'] = pickle.load(f)
            with open(itmconf_embeds_path, 'rb') as f:
                configs['itmconf_embeds'] = pickle.load(f)
            commconf_embeds_path = data_dir / 'community_conf_emb.pkl'
            with open(commconf_embeds_path, 'rb') as f:
                configs['commconf_embeds'] = pickle.load(f)
            return configs

        if model_name in {"discorec", "lightgcn_int"}:
            usrprf_embeds_path = data_dir / "usr_emb_np.pkl"
            itmprf_embeds_path = data_dir / "itm_emb_np.pkl"
            with open(usrprf_embeds_path, 'rb') as f:
                configs['usrprf_embeds'] = pickle.load(f)
            with open(itmprf_embeds_path, 'rb') as f:
                configs['itmprf_embeds'] = pickle.load(f)
            usrint_embeds_path = data_dir / "user_intent_emb_3.pkl"
            itmint_embeds_path = data_dir / "item_intent_emb_3.pkl"
            with open(usrint_embeds_path, 'rb') as f:
                configs['usrint_embeds'] = pickle.load(f)
            with open(itmint_embeds_path, 'rb') as f:
                configs['itmint_embeds'] = pickle.load(f)
            usrint_embeds_path = data_dir / "user_conf_emb.pkl"
            itmint_embeds_path = data_dir / "item_conf_emb.pkl"
            with open(usrint_embeds_path, 'rb') as f:
                configs['usrconf_embeds'] = pickle.load(f)
            with open(itmint_embeds_path, 'rb') as f:
                configs['itmconf_embeds'] = pickle.load(f)

        return configs

configs = parse_configure()

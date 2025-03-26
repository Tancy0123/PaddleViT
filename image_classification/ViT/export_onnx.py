import paddle
import paddle.nn as nn
import paddle2onnx
from vit import VisionTransformer
import numpy as np
import argparse
import os
from config import get_config
from config import update_config
from vit import build_vit as build_model

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None, help='Path to model config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to .pdparams file')
    parser.add_argument('--output', type=str, default='vit_model.onnx', help='Output ONNX file path')
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--batch_size_eval', type=int, default=None, help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--accum_iter', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resume training')
    parser.add_argument('--last_epoch', type=int, default=-1, help='Last epoch')
    parser.add_argument('--amp', action='store_true', help='Enable AMP')
    args = parser.parse_args()

    # Define model config
    # model_config = {
    #     'image_size': 224,
    #     'patch_size': 16,
    #     'in_channels': 3,
    #     'num_classes': 1000,
    #     'embed_dim': 768,
    #     'depth': 12,
    #     'num_heads': 12,
    #     'attn_head_size': None,
    #     'mlp_ratio': 4,
    #     'qkv_bias': True,
    #     'dropout': 0.,
    #     'attention_dropout': 0.,
    #     'droppath': 0.,
    #     'representation_size': None
    # }

    # Create model
    # model = VisionTransformer(**model_config)
    config = update_config(get_config(), args)

    model = build_model(config)
    # print("Model has been created")
    # for name, param in model.named_parameters():
    #     print(name)

    # print("Model printed")
    
    # Load pretrained weights
    if args.weights.endswith('.pdparams'):
        weights = paddle.load(args.weights)
        print(weights.keys())
        missing_keys = model.set_state_dict(weights['model'])
        print(f"Successfully loaded weights from {args.weights}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
    else:
        raise ValueError("Weights file must be a .pdparams file")

    # Set model to eval mode
    model.eval()

    # Create dummy input
    dummy_input = paddle.randn([1, 3, 224, 224])

    # Save model temporarily
    temp_model_path = args.output
    paddle.jit.save(
        layer=model,
        path=temp_model_path,
        input_spec=[dummy_input]
    )

    # Convert to ONNX
    save_path = args.output
    paddle2onnx.export(
        model_filename=temp_model_path + ".pdmodel",
        params_filename=temp_model_path + ".pdiparams",
        save_file=save_path,
        opset_version=11,
        enable_onnx_checker=True
    )
    
    # Clean up temporary files
    if os.path.exists(temp_model_path + ".pdmodel"):
        os.remove(temp_model_path + ".pdmodel")
    if os.path.exists(temp_model_path + ".pdiparams"):
        os.remove(temp_model_path + ".pdiparams")
    if os.path.exists(temp_model_path + ".pdiparams.info"):
        os.remove(temp_model_path + ".pdiparams.info")
    if os.path.exists(temp_model_path + ".json"):
        os.remove(temp_model_path + ".json")
        
    print(f"Model has been exported to {save_path}")

if __name__ == "__main__":
    main()

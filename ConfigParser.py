import argparse
import yaml


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"yes", "true", "t", "1", "y"}:
        return True
    if lowered in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")


def parse_sequence(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if text == "":
        return []
    parts = [item.strip() for item in text.split(",")]
    parsed = []
    for item in parts:
        lowered = item.lower()
        if lowered in {"true", "false", "yes", "no", "y", "n", "t", "f", "1", "0"}:
            parsed.append(str2bool(item))
            continue
        try:
            parsed.append(int(item))
            continue
        except ValueError:
            pass
        try:
            parsed.append(float(item))
            continue
        except ValueError:
            pass
        parsed.append(item)
    return parsed


def flatten_dict(cfg, parent_key="", sep="_"):
    flattened = {}
    for key, value in cfg.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, sep=sep))
        else:
            flattened[new_key] = value
    return flattened


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default = None, type = str, help = "Path to YAML config file")

    parser.add_argument('--dataset', default = 'ntu', type = str)
    parser.add_argument('--model_version', type = str)
    parser.add_argument('--model_type', default = 'deterministic', type=str)
    parser.add_argument('--num_joints', default = 17, type=int)
    parser.add_argument('--num_training_frame', default = 14, type = int)
    parser.add_argument('--num_testing_frame', default = 14, type = int)
    parser.add_argument('--num_epochs', default = 100, type = int)
    parser.add_argument('--batch_size', default = 16, type = int)
    parser.add_argument('--step_per_epoch', default = 200, type = int)
    parser.add_argument('--num_workers', default = 8, type = int)
    parser.add_argument('--pin_memory', default = True, type = str2bool)
    parser.add_argument('--decoupled_rep', default = False, type = str2bool)

    parser.add_argument('--device', default = 'cuda', type = str)
    parser.add_argument('--model', default = 'lstm', type = str)
    parser.add_argument('--model_input_type', default = None, type = str)
    parser.add_argument('--print_step', default = 50, type = int)
    parser.add_argument('--mode', default = 'baseline', type = str)
    parser.add_argument('--dropout', default = 0.2, type = float)
    parser.add_argument('--pck_alpha', default = 30, type = float) # 30 pixels for 2D images

    parser.add_argument('--frame_input_h', default = 1080, type = int)
    parser.add_argument('--frame_input_w', default = 1920, type = int)
    parser.add_argument('--boxsize', default = 640, type = int)

    parser.add_argument('--optimizer', default = 'Adam', type = str)
    parser.add_argument('--betas', default = (0.9, 0.999), type = parse_sequence)
    parser.add_argument('--temperature', default = 10000, type = float)
    parser.add_argument('--alpha', default = 1, type = float)
    parser.add_argument('--weight_decay', default = 1e-4, type = float)
    parser.add_argument('--teacher_forcing_rate', default = 0.0, type = float)
    parser.add_argument('--positional_encoding_temperature', default = 10000, type = float)
    parser.add_argument('--positional_encoding_alpha', default = 1, type = float)
    parser.add_argument('--max_norm', default = 0.1, type = float)

    parser.add_argument('--learning_rate', default = 1e-4, type = float)
    parser.add_argument('--warmup_learning_rate', default = 1e-6, type = float)
    parser.add_argument('--warmup_steps', default = 4000, type = int)
    parser.add_argument('--lr_scheduler', default = 'cosine', type = str)
    parser.add_argument('--lr_step_size', default = 1000, type = int)
    parser.add_argument('--lr_gamma', default = 0.2, type = float)
    parser.add_argument('--lr_eta_min', default = 1e-6, type = float)
    parser.add_argument('--lr_factor', default = 0.5, type = float)
    parser.add_argument('--lr_patience', default = 3, type = float)

    parser.add_argument('--loss_type', default = 'penalize', type = str) # naive, penalize, decoupled
    parser.add_argument('--lambda_norm', default = 2.0, type = float)
    parser.add_argument('--lambda_specs', default = 1.5, type = float)
    parser.add_argument('--lambda_root', default = 1.0, type = float)
    parser.add_argument('--root_pos', default = 0, type = int)
    parser.add_argument('--special_joints', default = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], type=parse_sequence)

    parser.add_argument('--mlp_hidden_layers', default = [1024, 2048, 1024], type = parse_sequence)

    parser.add_argument('--lstm_embedding_dim', default = 2048, type = int)
    parser.add_argument('--lstm_hidden_dim', default = 256, type = int)
    parser.add_argument('--lstm_num_layers', default = 2, type = int)
    parser.add_argument('--lstm_residual_connection', default = True, type = str2bool)

    parser.add_argument('--transformer_num_heads', default = 4, type = int)
    parser.add_argument('--transformer_model_dim', default = 256, type = int)
    parser.add_argument('--transformer_fc_hidden_dim', default = 2048, type = int)
    parser.add_argument('--transformer_num_layers', default = 4, type = int)
    parser.add_argument('--transformer_pre_normalization', default = False, type = str2bool)
    parser.add_argument('--transformer_residual_connection', default = True, type = str2bool)
    parser.add_argument('--transformer_type', default = 'vanilla', type = str)
    parser.add_argument('--get_attn_weight', default = False, type = str2bool)
    parser.add_argument('--joint_attn_embed_dim', default = 128, type = int)
    parser.add_argument('--joint_attn_num_layers', default = 2, type = int)
    parser.add_argument('--joint_attn_num_heads', default = 4, type = int)
    parser.add_argument('--joint_attn_fc_hidden_dim', default = 512, type = int)
    parser.add_argument('--proj_model_type', default = 'attn', type = str)

    parser.add_argument('--bert_num_heads', default = 4, type = int)
    parser.add_argument('--bert_num_layers', default = 6, type = int)
    parser.add_argument('--bert_model_dim', default = 256, type = int)
    parser.add_argument('--bert_fc_hidden_dim', default=1024, type = int)
    parser.add_argument('--bert_pre_normalization', default = False, type=str2bool)
    parser.add_argument('--bert_attention_mode', default = 'parallel', type = str)
    parser.add_argument('--bert_pose_encoder_type', default = 'mlp', type = str)
    parser.add_argument('--bert_pose_decoder_type', default = 'mlp', type = str)

    return parser


def parse_args():
    parser = get_parser()
    args, remaining = parser.parse_known_args()

    if args.config:
        with open(args.config, 'r') as cfg_file:
            cfg = yaml.safe_load(cfg_file) or {}
        parser.set_defaults(**flatten_dict(cfg))
        args = parser.parse_args(remaining)
    else:
        args = parser.parse_args()

    return args

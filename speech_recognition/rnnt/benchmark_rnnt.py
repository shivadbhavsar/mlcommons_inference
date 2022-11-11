import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "pytorch"))

import torch
import torch_migraphx
from decoders import ScriptGreedyDecoder
from model_separable_rnnt import RNNT
import toml
from helpers import add_blank_label

from torch_migraphx.fx.tools import mgx_benchmark


def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict


def load_model(config, checkpoint_path, rnnt_vocab):

    model = RNNT(feature_config=config['input_eval'],
                 rnnt=config['rnnt'],
                 num_classes=len(rnnt_vocab))

    model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path),
                          strict=False)

    model.eval()
    return model


if __name__ == '__main__':
    print('Loading Config...')
    checkpoint_path = '/code/Git/rnnt/rnnt.pt'
    config = toml.load('pytorch/configs/rnnt.toml')

    print('Loading Inputs and Modules...')
    feature, feature_length = torch.load('model_input.pt')
    mgx_encoder = torch.load('mgx_encoder.pt')
    mgx_prediction = torch.load('mgx_prediction.pt')

    dataset_vocab = config['labels']['labels']
    rnnt_vocab = add_blank_label(dataset_vocab)

    print('Loading Model...')
    model = load_model(config, checkpoint_path, rnnt_vocab)

    model, model.encoder, model.prediction, model.joint = model.cuda(
    ), model.encoder.cuda(), model.prediction.cuda(), model.joint.cuda()
    feature, feature_length = feature.cuda(), feature_length.cuda()
    greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, model)

    model_mgx = load_model(config, checkpoint_path, rnnt_vocab)
    model_mgx, model_mgx.joint = model_mgx.cuda(), model_mgx.joint.cuda()
    model_mgx.encoder, model_mgx.prediction = mgx_encoder, mgx_prediction
    greedy_decoder_mgx = ScriptGreedyDecoder(len(rnnt_vocab) - 1, model_mgx)

    print('Running Torch Model')
    _, _, original_transcript = greedy_decoder(feature, feature_length)
    print(original_transcript)

    print('Running MGX Model')
    _, _, mgx_transcript = greedy_decoder_mgx(feature, feature_length)
    print(mgx_transcript)

    print('Benchmarking Torch Model')
    mgx_benchmark.benchmark(greedy_decoder, [feature, feature_length],
                            batch_size=1)

    print('Benchmarking MGX Model')
    mgx_benchmark.benchmark(greedy_decoder_mgx, [feature, feature_length],
                            batch_size=1)

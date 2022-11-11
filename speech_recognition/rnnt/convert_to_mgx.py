import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "pytorch"))

import torch
import numpy as np
import toml
from helpers import add_blank_label

from model_separable_rnnt import RNNT
from QSL import AudioQSL, AudioQSLInMemory
from preprocessing import AudioPreprocessing
from decoders import ScriptGreedyDecoder
from rnn import StackTime
from model_separable_rnnt import Embed

import torch_migraphx
from torch_migraphx.fx.utils import LowerPrecision


def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict


if __name__ == '__main__':
    dataset_dir = '/code/Git/rnnt/local_data/'
    manifest_filepath = dataset_dir + 'dev-clean-wav.json'

    print('Loading Config...')
    checkpoint_path = '/code/Git/rnnt/rnnt.pt'
    config = toml.load('pytorch/configs/rnnt.toml')
    dataset_vocab = config['labels']['labels']
    rnnt_vocab = add_blank_label(dataset_vocab)
    # print(rnnt_vocab)
    featurizer_config = config['input_eval']
    # print(featurizer_config)

    print('Initializing Models...')
    model = RNNT(feature_config=featurizer_config,
                 rnnt=config['rnnt'],
                 num_classes=len(rnnt_vocab))
    model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path),
                          strict=False)
    model.eval()

    print('Loading Data...')
    qsl = AudioQSLInMemory(dataset_dir,
                           manifest_filepath,
                           dataset_vocab,
                           featurizer_config["sample_rate"],
                           perf_count=10)

    print('Initializing Audio Preprocessing...')
    audio_preprocessor = AudioPreprocessing(**featurizer_config)
    audio_preprocessor.eval()
    audio_preprocessor = torch.jit.script(audio_preprocessor)
    audio_preprocessor = torch.jit._recursive.wrap_cpp_module(
        torch._C._freeze_module(audio_preprocessor._c))

    waveform = qsl[1]
    waveform_length = np.array(waveform.shape[0], dtype=np.int64)
    waveform = np.expand_dims(waveform, 0)
    waveform_length = np.expand_dims(waveform_length, 0)
    with torch.no_grad():
        waveform = torch.from_numpy(waveform)
        waveform_length = torch.from_numpy(waveform_length)
        feature, feature_length = audio_preprocessor.forward(
            (waveform, waveform_length))
        feature = feature.permute(2, 0, 1)

        model, model.encoder, model.prediction, model.joint = model.cuda(
        ), model.encoder.cuda(), model.prediction.cuda(), model.joint.cuda()
        feature, feature_length = feature.cuda(), feature_length.cuda()

        torch.save([feature, feature_length], 'model_input.pt')

        print('Running Conversion...')
        # mgx_mod = torch_migraphx.fx.lower_to_mgx(
        #     greedy_decoder, [feature, feature_length],
        #     min_acc_module_size=1,
        #     lower_precision=LowerPrecision.FP16,
        #     verbose_log=True)
        mgx_encoder = torch_migraphx.fx.lower_to_mgx(
            model.encoder,
            [feature, feature_length],
            min_acc_module_size=1,
            lower_precision=LowerPrecision.FP16,
            verbose_log=True,
            leaf_modules=[StackTime],
            save_subgraph_programs=True,
        )
        torch.save(mgx_encoder, 'mgx_encoder.pt')
        print('-' * 50)

        _, h0_sample_in, c0_sample_in = model.prediction.cf_embed(None)
        label = torch.tensor([[0]], dtype=torch.int64).cuda()
        mgx_prediction = torch_migraphx.fx.lower_to_mgx(
            model.prediction, [label, (h0_sample_in, c0_sample_in)],
            min_acc_module_size=1,
            lower_precision=LowerPrecision.FP16,
            verbose_log=True,
            leaf_modules=[Embed])
        torch.save(mgx_prediction, 'mgx_prediction.pt')

        print('Loading Models')
        mgx_encoder = torch.load('mgx_encoder.pt')
        mgx_prediction = torch.load('mgx_prediction.pt')

        greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, model)
        _, _, original_transcript = greedy_decoder(feature, feature_length)

        print(original_transcript)

        model.encoder = mgx_encoder
        model.prediction = mgx_prediction
        mgx_greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, model)
        _, _, mgx_transcript = mgx_greedy_decoder(feature, feature_length)
        print(mgx_transcript)
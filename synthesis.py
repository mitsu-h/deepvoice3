# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesis.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --checkpoint-seq2seq=<path>       Load seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>       Load postnet model from checkpoint path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --max-decoder-steps=<N>           Max decoder steps [default: 500].
    --replace_pronunciation_prob=<N>  Prob [default: 0.5].
    --speaker_id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    --type=<s>                        vocoder tyoe [default: linear]
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext

import audio

import torch
import numpy as np
import nltk
import time
#nltk.download('punkt')

# The deepvoice3 model
from deepvoice3_pytorch import frontend
from hparams import hparams, hparams_debug_string

from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_frontend = None  # to be set later


def tts(model, text, p=0, speaker_id=None, fast=False):
    """Convert text to speech waveform given a deepvoice3 model.

    Args:
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    """
    model = model.to(device)
    model.eval()
    if fast:
        model.make_generation_fast_()

    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)
    speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(device)

    # Greedy decoding
    with torch.no_grad():
        mel_outputs, vocoder_parameter, alignments, done = model(
            sequence, text_positions=text_positions, speaker_ids=speaker_ids)

    alignment = alignments[0].cpu().data.numpy()
    mel = mel_outputs[0].cpu().data.numpy()
    mel = audio._denormalize(mel)
    if type(vocoder_parameter) is tuple:
        _, f0s, sps, aps = vocoder_parameter
        f0 = f0s[0].cpu().data.numpy() * 400
        sp = sps[0].cpu().data.numpy()
        ap = aps[0].cpu().data.numpy()
        waveform = audio.world_synthesize(f0, sp, ap)
        spectrogram = (f0, sp, ap)
    else:
        linear_output = vocoder_parameter[0].cpu().data.numpy()
        spectrogram = audio._denormalize(linear_output)
        # Predicted audio signal
        waveform = audio.inv_spectrogram(linear_output.T)


    return waveform, alignment, spectrogram, mel


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    checkpoint_seq2seq_path = args["--checkpoint-seq2seq"]
    checkpoint_postnet_path = args["--checkpoint-postnet"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]
    replace_pronunciation_prob = float(args["--replace_pronunciation_prob"])
    output_html = args["--output-html"]
    speaker_id = args["--speaker_id"]
    training_type = args["--type"]
    if speaker_id is not None:
        speaker_id = int(speaker_id)
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    _frontend = getattr(frontend, hparams.frontend)
    import training_module as tm
    tm._frontend = _frontend
    from training_module import plot_alignment, build_model

    # Model
    model = build_model(training_type=training_type)

    # Load checkpoints separately
    if checkpoint_postnet_path is not None and checkpoint_seq2seq_path is not None:
        checkpoint = _load(checkpoint_seq2seq_path)
        model.seq2seq.load_state_dict(checkpoint["state_dict"])
        checkpoint = _load(checkpoint_postnet_path)
        model.postnet.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_seq2seq_path))[0]
    else:
        checkpoint = _load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_path))[0]

    model.seq2seq.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)
    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines, 1):
            text = line.decode("utf-8")[:-1]
            words = nltk.word_tokenize(text)
            start = time.time()
            waveform, alignments, _, mel = tts(
                model, text, p=replace_pronunciation_prob, speaker_id=speaker_id, fast=True)
            end = time.time() - start
            dst_wav_path = join(dst_dir, "{}_{}{}.wav".format(
                idx, checkpoint_name, file_name_suffix))
            dst_world_path = join(dst_dir, "{}_{}{}_world.wav".format(
                idx, checkpoint_name, file_name_suffix))
            for i, alignment in enumerate(alignments, 1):
                dst_alignment_path = join(
                    dst_dir, "{}_{}{}_alignment_layer_{}.png".format(idx, checkpoint_name,
                                                        file_name_suffix,i))
                plot_alignment(alignment.T, dst_alignment_path,
                               info="{}, {}, layer_{}".format(hparams.builder, basename(checkpoint_path),i))
            audio.save_wav(waveform, dst_wav_path)
            name = splitext(basename(text_list_file_path))[0]
            if output_html:
                print("""
{}

({} chars, {} words)

<audio controls="controls" >
<source src="/audio/{}/{}/{}" autoplay/>
Your browser does not support the audio element.
</audio>

<div align="center"><img src="/audio/{}/{}/{}" /></div>
                  """.format(text, len(text), len(words),
                             hparams.builder, name, basename(dst_wav_path),
                             hparams.builder, name, basename(dst_alignment_path)))
            else:
                print(idx, ": {}\n ({} chars, {} words) generate time:{}s".format(text, len(text), len(words), end))

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)

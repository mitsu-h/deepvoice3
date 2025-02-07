from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
from hparams import hparams

import time


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            if len(text) < hparams.min_text:
                continue
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, index, wav_path, text)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # Load the audio to a numpy array:

    wav = audio.load_wav(wav_path)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max



    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    
    #world parameters
    f0,sp,ap = audio.world(wav,hparams.sample_rate)
    f0 = (f0 / hparams.f0_norm).astype(np.float32) #normalize
    sp = audio._normalize(sp).astype(np.float32)
    ap = ap.astype(np.float32) #apは0~1の範囲しか値を取らないので正規化不要
    world_frames = f0.shape[0]
    




    # Write the spectrograms to disk:
    spectrogram_filename = 'ljspeech-spec-%05d.npy' % index
    mel_filename = 'ljspeech-mel-%05d.npy' % index
    
    f0_filename = 'ljspeech-f0-%05d.npy' % index
    sp_filename = 'ljspeech-sp-%05d.npy' % index
    ap_filename = 'ljspeech-ap-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, f0_filename), f0, allow_pickle=False)
    np.save(os.path.join(out_dir, sp_filename), sp, allow_pickle=False)
    np.save(os.path.join(out_dir, ap_filename), ap, allow_pickle=False)
    


    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, f0_filename, sp_filename, ap_filename, world_frames, text)
    '''
    audio_filename = 'ljspeech-spec-%05d.npy' % index
    np.save(os.path.join(out_dir, audio_filename), wav, allow_pickle=False)

    return (audio_filename, wav.shape[0], text)
    '''

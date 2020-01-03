"""
usage: load_test.py <in_dir> <out_dir>

"""

from docopt import docopt
import numpy as np
from train import LinearSpecDataSource, SpDataSource, MelSpecDataSource,FileSourceDataset
import os
from os.path import join
from tqdm import tqdm
import time
import audio

if __name__ == '__main__':
    args = docopt(__doc__)
    print("Command line args:\n", args)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]

    speaker_id = None
    SP = FileSourceDataset(SpDataSource(in_dir, speaker_id))
    for i, sp in tqdm(enumerate(SP,1)):
        log_sp = audio._amp_to_db(np.abs(sp)) -10
        log_sp = audio._normalize(log_sp)
        sp_filename = 'ljspeech-sp-%05d.npy' % i
        np.save(os.path.join(out_dir, sp_filename), log_sp, allow_pickle=False)

    '''
    os.makedirs(out_dir, exist_ok=True)
    
    speaker_id = None
    SP = FileSourceDataset(SpDataSource(in_dir, speaker_id))
    AP = FileSourceDataset(ApDataSource(in_dir,speaker_id))

    for i, (sp,ap) in enumerate(zip(SP,AP),start=1):
        spap = np.stack((sp,ap))
        spap_filename = 'ljspeech-spap-%05d.npy' % i
        np.save(os.path.join(out_dir, spap_filename), spap, allow_pickle=False)
        
        meta = join(in_dir, "train.txt")
    with open(meta, "rb") as f:
        lines = f.readlines()
    l = [lines[i].decode("utf-8").split("|") for i in range(len(lines))]
    with open(os.path.join(in_dir, 'train_new.txt'), 'w', encoding='utf-8') as f:
        for tex in l:
            tex.pop(5)
            f.write('|'.join([str(x) for x in tex]))
    '''


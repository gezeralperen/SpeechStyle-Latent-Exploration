import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import argparse
import os
import time
import IPython
import pyworld as pw

from styler import STYLER
from dataset import Dataset
from evaluate import evaluate
from synthesize import preprocess_text, synthesize, read, preprocess_audio, get_processed_data_from_wav
import hparams as hp
import utils
import audio as Audio

torch.manual_seed(0)

# Get device
device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

# Get dataset
dataset = Dataset("train.txt")
loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True,
                    collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)

# Define model
model = nn.DataParallel(STYLER()).to(device)

checkpoint_path = os.path.join(hp.checkpoint_path())
checkpoint = torch.load(os.path.join(
    checkpoint_path, 'checkpoint_1190000.pth.tar'))
model.load_state_dict(checkpoint['model'])

model.requires_grad = False
model.eval()

# Load vocoder
vocoder = utils.get_vocoder()


#############################################################

reference = 'Evelynn_001'

speaker_id = 'Evelynn'

target_sentence = 'Hello! I am going to tell you something really cool! Are you ready?'

##############################################################

audio_path = f'wav_data\\{speaker_id}\\{reference}.wav'
tg_path = f'preprocessed\\VCTK\\TextGrid\\{speaker_id}\\{reference}.TextGrid'

text = preprocess_text(target_sentence)

spker_embed_path = os.path.join(
                hp.preprocessed_path, "spker_embed", "{}-spker_embed-{}.npy".format(hp.dataset, speaker_id))
speaker_embed = torch.from_numpy(np.load(spker_embed_path)).to(device)

_, wav = read(audio_path)


f0, energy, mel = get_processed_data_from_wav(audio_path, tg_path, False)

energy = (energy-hp.energy_min)/(hp.energy_max-hp.energy_min)
f0_norm = utils.speaker_normalization(f0)
mel, mel_len, energy, f0, f0_norm = preprocess_audio(mel, energy, f0, f0_norm)

with torch.no_grad():
    output = synthesize('', model, vocoder, text, target_sentence, speaker_embed, speaker_id, False, mel, mel_len, f0, f0_norm, energy, write=False)
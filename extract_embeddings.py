from cmath import pi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import os
import time

from styler import STYLER
from dataset import Dataset
import hparams as hp
import utils

import sqlite3

if os.path.exists("embeddings_vctk.db"):
  os.remove("embeddings_vctk.db")
if os.path.exists("embeddings_vctk.db-journal"):
  os.remove("embeddings_vctk.db-journal")

connection = sqlite3.connect('embeddings_vctk.db')
crsr = connection.cursor()

init_tables =  "CREATE TABLE embeddings ( num INT, " + ', '.join([ f'col{i} FLOAT(7,5)' for i in range(256)]) + ');'
crsr.execute(init_tables)

def extract_encodings(model, src_seq, mel_target, mel_aug, p_norm, e_input, src_len, mel_len, d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None, speaker_embed=None, d_control=1.0, p_control=1.0, e_control=1.0):
    with torch.no_grad():
        src_mask = utils.get_mask_from_lengths(src_len, max_src_len)
        mel_mask = utils.get_mask_from_lengths(mel_len, max_mel_len)
        (text_encoding, pitch_embedding, speaker_encoding, energy_embedding), noise_encoding, d_prediction, p_prediction, e_prediction, mel_len, mel_mask, (aug_posterior_d, aug_posterior_p, aug_posterior_e) = model.module.style_modeling(
                        src_seq, speaker_embed, mel_target, mel_aug, p_norm, e_input, src_len, mel_len, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len, d_control, p_control, e_control, seperate=True)
        return (text_encoding, pitch_embedding, speaker_encoding, energy_embedding), noise_encoding, d_prediction, p_prediction, e_prediction, mel_len, mel_mask, (aug_posterior_d, aug_posterior_p, aug_posterior_e)


torch.manual_seed(0)

# Get device
device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')


# Define model
model = nn.DataParallel(STYLER()).to(device)

checkpoint_path = os.path.join(hp.checkpoint_path())
checkpoint = torch.load(os.path.join(
    checkpoint_path, 'checkpoint_300000.pth.tar'))
model.load_state_dict(checkpoint['model'])

# Load vocoder
vocoder = utils.get_vocoder()

model.requires_grad = False
model.eval()

dataset = Dataset("train.txt")

len = len(dataset)

num = 0
for i, data_of_batch in enumerate(dataset):
    # Get Data
    text = torch.from_numpy(
        data_of_batch["text"]).long().to(device).unsqueeze(0)
    mel_target = torch.from_numpy(
        data_of_batch["mel_target"]).float().to(device).unsqueeze(0)
    mel_aug = torch.from_numpy(
        data_of_batch["mel_aug"]).float().to(device).unsqueeze(0)
    D = torch.from_numpy(data_of_batch["D"]).long().to(device).unsqueeze(0)
    log_D = torch.from_numpy(np.log(D.detach().cpu().numpy() + hp.log_offset)).float().to(device).unsqueeze(0)
    f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device).unsqueeze(0)
    f0_norm = torch.from_numpy(data_of_batch["f0_norm"]).float().to(device).unsqueeze(0)
    f0_norm_aug = torch.from_numpy(data_of_batch["f0_norm_aug"]).float().to(device).unsqueeze(0)
    energy = torch.from_numpy(
        data_of_batch["energy"]).float().to(device).unsqueeze(0)
    energy_input = torch.from_numpy(
        data_of_batch["energy_input"]).float().to(device).unsqueeze(0)
    energy_input_aug = torch.from_numpy(
        data_of_batch["energy_input_aug"]).float().to(device).unsqueeze(0)
    speaker_embed = torch.from_numpy(
        data_of_batch["speaker_embed"]).float().to(device)
    src_len = torch.from_numpy(np.array([text.shape[1]])).long().to(device)
    mel_len = torch.from_numpy(
        np.array([mel_target.shape[1]])).long().to(device)
    max_src_len = src_len.detach().cpu().numpy()[0]
    max_mel_len = mel_len.detach().cpu().numpy()[0]
    
    (text_encoding, pitch_embedding, speaker_encoding, energy_embedding), noise_encoding, d_prediction, p_prediction, e_prediction, mel_len, mel_mask, (aug_posterior_d, aug_posterior_p, aug_posterior_e) = extract_encodings(model, text, mel_target, mel_aug, f0_norm, energy_input, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len, speaker_embed=speaker_embed)
    
    
    arr1 = (pitch_embedding+energy_embedding).cpu().detach().numpy()
    for x in range(arr1[0].shape[0]):
        if num % 100 == 0:
            st1 = f"INSERT INTO embeddings \nVALUES({num}, " + ', '.join(['%.5f' % num for num in arr1[0, x]]) + ");"
            crsr.execute(st1)
            connection.commit()
        num += 1
    print(f'{100*i/len:.3f}', end='\r')

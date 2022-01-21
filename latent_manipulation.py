import pandas as pd
import sqlite3
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hparams as hp
import utils
from styler import STYLER
import os
import numpy as np
from synthesize import preprocess_text, synthesize, read, preprocess_audio, get_processed_data_from_wav
import IPython

torch.manual_seed(0)

# Get device
device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')


# Define model
model = nn.DataParallel(STYLER()).to(device)

checkpoint_path = os.path.join(hp.checkpoint_path())
checkpoint = torch.load(os.path.join(
    checkpoint_path, 'checkpoint_1190000.pth.tar'))
model.load_state_dict(checkpoint['model'])

# Load vocoder
vocoder = utils.get_vocoder()

model.requires_grad = False
model.eval()

reference = 'Evelynn_002'

speaker_id = 'Evelynn'

target_speaker = 'Jinx'

speaker_example = f'wav_data\\{target_speaker}\\{target_speaker}_001.wav'

audio_path = f'wav_data\\{speaker_id}\\{reference}.wav'
tg_path = f'preprocessed\\VCTK\\TextGrid\\{speaker_id}\\{reference}.TextGrid'

target_sentence = utils.get_transcript(f'wav_data\\{speaker_id}\\{reference}.txt')

text = preprocess_text(target_sentence)

spker_embed_path = os.path.join(
                hp.preprocessed_path, "spker_embed", "{}-spker_embed-{}.npy".format(hp.dataset, speaker_id))
speaker_embed = torch.from_numpy(np.load(spker_embed_path)).to(device)

target_spker_embed_path = os.path.join(
                hp.preprocessed_path, "spker_embed", "{}-spker_embed-{}.npy".format(hp.dataset, target_speaker))
target_speaker_embed = torch.from_numpy(np.load(target_spker_embed_path)).to(device)

_, wav = read(audio_path)


f0, energy, mel = get_processed_data_from_wav(audio_path, tg_path, False)

energy = (energy-hp.energy_min)/(hp.energy_max-hp.energy_min)
f0_norm = utils.speaker_normalization(f0)
mel, mel_len, energy, f0, f0_norm = preprocess_audio(mel, energy, f0, f0_norm)

connection = sqlite3.connect('embeddings.db')
embeddings = pd.read_sql_query("SELECT * from embeddings", connection)
embeddings.drop('num', inplace=True, axis=1)

pca = PCA()
pca.fit(embeddings)

latent_directions = pca.components_[:6]
latent_directions = torch.from_numpy(np.array(latent_directions)).to(device).type(torch.float32)

def extract_encodings(model, src_seq, mel_target, mel_aug, p_norm, e_input, src_len, mel_len, d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None, speaker_embed=None, d_control=1.0, p_control=1.0, e_control=1.0):
    with torch.no_grad():
        src_mask = utils.get_mask_from_lengths(src_len, max_src_len)
        mel_mask = utils.get_mask_from_lengths(mel_len, max_mel_len)
        (text_encoding, pitch_embedding, speaker_encoding, energy_embedding), noise_encoding, d_prediction, p_prediction, e_prediction, mel_len, mel_mask, (aug_posterior_d, aug_posterior_p, aug_posterior_e) = model.module.style_modeling(
                        src_seq, speaker_embed, mel_target, mel_aug, p_norm, e_input, src_len, mel_len, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len, d_control, p_control, e_control, seperate=True)
        return (text_encoding, pitch_embedding, speaker_encoding, energy_embedding), noise_encoding, d_prediction, p_prediction, e_prediction, mel_len, mel_mask, (aug_posterior_d, aug_posterior_p, aug_posterior_e)

alpha = 0.5

src_len = torch.from_numpy(np.array([text.shape[1]])).long().to(device)
(text_encoding, pitch_embedding, speaker_encoding, energy_embedding), noise_encoding, d_prediction, p_prediction, e_prediction, mel_len, mel_mask, (aug_posterior_d, aug_posterior_p, aug_posterior_e) = extract_encodings(model, text, mel, mel, f0_norm, energy, src_len, mel_len,speaker_embed=speaker_embed)
style_modeling_output = text_encoding + pitch_embedding + speaker_encoding + energy_embedding

for i, direction in enumerate(latent_directions):
    print(f"Direction +-{i}")
    style = style_modeling_output + alpha * direction

    mel_output, mel_output_postnet = model.module.decode(style, mel_mask)
    mel_postnet_torch = mel_output_postnet.transpose(1, 2).detach()
    mel_output_postnet = mel_output_postnet[0].cpu().transpose(0, 1).detach()

    output = utils.vocoder_infer(mel_postnet_torch, vocoder, '', write=False)
    IPython.display.display(IPython.display.Audio(output, rate=22050))


    style = style_modeling_output - alpha * direction

    mel_output, mel_output_postnet = model.module.decode(style, mel_mask)
    mel_postnet_torch = mel_output_postnet.transpose(1, 2).detach()
    mel_output_postnet = mel_output_postnet[0].cpu().transpose(0, 1).detach()

    output = utils.vocoder_infer(mel_postnet_torch, vocoder, '', write=False)
    IPython.display.display(IPython.display.Audio(output, rate=22050))
    
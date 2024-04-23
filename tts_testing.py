import torch
from TTS.api import TTS
import librosa as ls
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

def SynthesizeTTS(model_name_, text_, filename):
    # Get device
    device = "cpu"
    gen_fpath = "/mnt/data/" + filename
    ref_fpath = "/mnt/data/ref_" + filename
    
    tts = TTS(model_name=model_name_, progress_bar=True).to(device)
    tts.tts_to_file(text=text_, file_path=gen_fpath)
    
    ref_wav, ref_sr = ls.load(ref_fpath, sr=44400)
    gen_wav, gen_sr = ls.load(gen_fpath, sr=44400)
    
    # if (ref_wav.shape[0] != gen_wav.shape[0]):
    #     print("RESHAPED!!!")
    #     mshape = max(ref_wav.shape[0], gen_wav.shape[0])
    #     if (ref_wav.shape[0] == mshape):
    #         add = mshape - gen_wav.shape[0]
    #         gen_wav = np.pad(gen_wav, (0, add), mode="constant", constant_values=0)
    #     else:
    #         add = mshape - ref_wav.shape[0]
    #         ref_wav = np.pad(ref_wav, (0, add), mode="constant", constant_values=0)
    
    chroma_ref = ls.feature.chroma_stft(y=ref_wav, sr=ref_sr, hop_length=1024)
    chroma_gen = ls.feature.chroma_stft(y=gen_wav, sr=gen_sr, hop_length=1024)
    
    # chroma_ref = ls.feature.chroma_cqt(y=ref_wav, sr=ref_sr, hop_length=1024)
    # chroma_gen = ls.feature.chroma_cqt(y=gen_wav, sr=gen_sr, hop_length=1024)
    
    x_ref = ls.feature.stack_memory(chroma_ref, n_steps=10, delay=3)
    x_gen = ls.feature.stack_memory(chroma_gen, n_steps=10, delay=3)
    
    xsim =ls.segment.cross_similarity(x_ref, x_gen, metric="cosine")
    
    plt.figure(figsize = (10,7))
    imgsim = ls.display.specshow(xsim, x_axis='s', y_axis='s', hop_length=1024)
    results_path = "/mnt/data/" + filename + '.png'
    plt.savefig(results_path, dpi=400)


# List available 🐸TTS models
print("Available models:")
tts_manager = TTS().list_models()
all_models = tts_manager.list_models()
for model in all_models:
    print(model)
print("------------------------")
print("Beginning the tests:\n")

print("Creating test for en")
SynthesizeTTS("tts_models/en/ljspeech/tacotron2-DDC", "Checking English language synthesis", "en.wav")
print("Creating test for de")
SynthesizeTTS("tts_models/de/thorsten/tacotron2-DDC", "überprüfung der englischen Synthese", "de.wav")
print("Creating test for it")
SynthesizeTTS("tts_models/it/mai_male/vits", "verifica della seintesi della lingua inglese", "it.wav")
print("Testing done!")

import torch
from TTS.api import TTS

def SynthesizeTTS(model_name_, text_, filename):
    # Get device
    device = "cpu"
    tts = TTS(model_name=model_name_, progress_bar=True).to(device)
    fpath = "/mnt/data/" + filename
    tts.tts_to_file(text=text_, file_path=fpath)


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

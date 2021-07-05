import librosa
import librosa.display
from IPython.display import Audio
from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
import torch
from PIL import Image


rate = 22050


def load_img(path, size=224):
    img = image.imread(path)
    img = np.array(Image.fromarray(img).resize([size, size]))
    img = torch.tensor(img).to('cuda') / 255.
    return img[None]


def imshow(img):
    plt.imshow(img.cpu().detach().numpy().squeeze())
    plt.axis('off')
    plt.show()


def load_audio(filename):
    data, _ = librosa.load(filename, sr=rate)
    return torch.tensor(data.data).reshape(-1, 1).cuda()


def play(audio):
    if not isinstance(audio, np.ndarray):
        audio = audio.cpu().detach().numpy()
    audio = audio.squeeze()
    display(Audio(audio, rate=rate))
    hop_length = 1024
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=hop_length)),
                                ref=np.max)
    specplot = librosa.display.specshow(D, y_axis='log', sr=rate, hop_length=hop_length,
                                        x_axis='time', ax=ax)
    ax.set(title='Log-frequency power spectrogram')
    ax.label_outer()
    fig.colorbar(specplot, ax=ax, format="%+2.f dB")
    plt.show()

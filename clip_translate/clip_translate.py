import torch
from model import AudioCLIP
import numpy as np
from siren_pytorch import SirenNet, SirenWrapperNDim
from torch import nn
from clip_translate.utils import play
from matplotlib import pyplot as plt


rate = 22050  # required by AudioCLIP


def get_perceptor(pretrained):
    torch.set_grad_enabled(False)
    perceptor = AudioCLIP(pretrained=pretrained).cuda()
    perceptor.eval()
    torch.set_grad_enabled(True)
    return perceptor


class AudioImagine():
    def __init__(self,
                 perceptor,
                 image=None,
                 text=None):
        if isinstance(perceptor, str):
            perceptor = get_perceptor(perceptor)
        self.perceptor = perceptor
        if image is not None:
            self.image_enc = self.encode_img(image).detach()
        else:
            self.image_enc = None
        if text is not None:
            self.text_enc = self.encode_text(text).detach()
        else:
            self.text_enc = None

    def encode_text(self, text):
        with torch.no_grad():
            text_enc = self.perceptor.encode_text(text)
            text_enc = text_enc / text_enc.norm(dim=-1, keepdim=True)
            return text_enc

    def encode_img(self, image):
        with torch.no_grad():
            image = image.permute(0, 3, 1, 2)
            img_enc = self.perceptor.encode_image(image)
            img_enc = img_enc / img_enc.norm(dim=-1, keepdim=True)
            return img_enc

    def augment_audio(self, audio, min_seconds=1, max_seconds=4):
        # random crops
        audio = self.to_audio_shape(audio)
        seconds = np.random.uniform(min_seconds, max_seconds)
        frames = int(seconds * rate)
        cutoff = audio.shape[1] - frames
        cutoff_start = np.random.randint(0, cutoff)
        cutoff_end = cutoff - cutoff_start
        audio = audio[:, cutoff_start:-cutoff_end]

        if np.random.rand() > 0.5:
            audio = - audio

        if np.random.rand() > 0.5:
            audio = audio * np.random.uniform(0.3, 1.0)

        return audio

    def to_audio_shape(self, audio):
        audio = audio.squeeze()[None]
        # if len(audio.shape) == 3:
        #     audio = audio[:, :, 0]
        # if len(audio.shape) == 1:
        #     audio = audio[None]
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio)
        return audio

    def encode_audio(self, audio, augment=False):
        if augment:
            audio = self.augment_audio(audio)
        else:
            audio = self.to_audio_shape(audio)

        audio = audio * 32768.0
        audio_enc = self.perceptor.encode_audio(audio)
        audio_enc = audio_enc / audio_enc.norm(dim=-1, keepdim=True)
        return audio_enc

    def encode_text(self, text):
        with torch.no_grad():
            text_enc = self.perceptor.encode_text([text.split()])
            text_enc = text_enc / text_enc.norm(dim=-1, keepdim=True)
            return text_enc

    def get_score(self, audio, augment=True):
        # return torch.sum(audio)
        audio_enc = self.encode_audio(audio, augment=augment)
        score = 0
        if self.image_enc is not None:
            score += torch.cosine_similarity(audio_enc,
                                             self.image_enc, -1).mean()
        if self.text_enc is not None:
            score += torch.cosine_similarity(audio_enc,
                                             self.text_enc, -1).mean()
        return score


def get_audio_siren():
    return SirenNet(
        dim_in=1,
        dim_hidden=256,
        dim_out=1,
        num_layers=3,
        w0=30.,
        w0_initial=3000.,
        use_bias=True,
        final_activation=None)


def get_siren_decoder(output_shape, latent_dim=1024):
    net = get_audio_siren()

    decoder = SirenWrapperNDim(
        net,
        latent_dim=latent_dim,
        output_shape=output_shape
    )
    decoder.cuda()

    return decoder


def fit_siren(imagine, siren, latent=None, steps=2000):
    optim = torch.optim.Adam(lr=1e-4, params=siren.parameters())
    steps_till_summary = 1000
    for step in range(steps):
        model_output = siren(latent=latent) 
        loss = -1 * imagine.get_score(model_output, augment=True)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % steps_till_summary == 0:
            print(loss.cpu().detach())
            pred_audio = siren(latent = latent)
            play(pred_audio)
            plt.plot(pred_audio.cpu().detach().numpy().squeeze())
            plt.show()

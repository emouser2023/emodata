import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torchvision.models import mobilenet_v2

# class AudioEncoder(nn.Module):
#     def __init__(self, output_dim=512, sample_rate=16000, n_mels=64, max_audio_length=15.0):
#         super(AudioEncoder, self).__init__()
#         self.sample_rate = sample_rate
#         self.max_samples = int(max_audio_length * sample_rate)
        
#         # Mel-spectrogram transformation
#         self.mel_spec = T.MelSpectrogram(
#             sample_rate=sample_rate,
#             n_fft=1024,
#             hop_length=256,
#             n_mels=n_mels
#         )
#         self.to_db = T.AmplitudeToDB()
        
#         # Pretrained MobileNetV2 backbone (lightweight)
#         mobilenet = mobilenet_v2(pretrained=True)
#         # Modify first conv layer to accept 1 channel (spectrogram) instead of 3 (RGB)
#         mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
#         self.cnn = mobilenet.features  # Use feature extractor only
        
#         # Projection head
#         self.fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
#             nn.Flatten(),
#             nn.Linear(1280, 512),  # MobileNetV2 output is 1280-dim
#             nn.ReLU(),
#             nn.Linear(512, output_dim)
#         )
    
#     def forward(self, audio):
#         audio = self._preprocess_audio(audio)
#         spec = self.mel_spec(audio)
#         spec = self.to_db(spec)
#         spec = spec.unsqueeze(1)  # (batch_size, 1, n_mels, time)
        
#         features = self.cnn(spec)  # (batch_size, 1280, H', W')
#         embedding = self.fc(features)  # (batch_size, 512)
#         embedding = F.normalize(embedding, dim=-1)
#         return embedding
    
#     def _preprocess_audio(self, audio):
#         batch_size = audio.size(0)
#         audio_len = audio.size(1)
#         if audio_len > self.max_samples:
#             audio = audio[:, :self.max_samples]
#         elif audio_len < self.max_samples:
#             padding = self.max_samples - audio_len
#             audio = torch.nn.functional.pad(audio, (0, padding))
#         return audio
    


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

class AudioEncoder_1(nn.Module):
    def __init__(self, output_dim=512, sample_rate=16000, n_mels=64, max_audio_length=15.0):
        super(AudioEncoder_1, self).__init__()
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)
        
        # Mel-spectrogram transformation
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            f_min=50.0,
            f_max=14000.0
        )
        self.to_db = T.AmplitudeToDB(top_db=80.0)
        
        # Use a pretrained model from torchaudio.pipelines (e.g., HuBERT_BASE)
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.cnn = bundle.get_model()  # Pretrained HuBERT Base (~94M params, lighter than large variants)
        
        # Freeze pretrained weights
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # HuBERT outputs 768-dim features, so project to 512
        self.fc = nn.Sequential(
            nn.Linear(768, 512),  # Adjust input dim based on model output
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, audio):
        audio = self._preprocess_audio(audio)
        
        # HuBERT expects raw waveforms, not spectrograms
        with torch.no_grad():  # Inference mode for pretrained model
            features, _ = self.cnn(audio)  # (batch_size, seq_len, 768)
            features = features.mean(dim=1)  # Average over time: (batch_size, 768)
        # features, _ = self.cnn(audio)  # (batch_size, seq_len, 768)
        # features = features.mean(dim=1)  
        
        # Project to 512-dim embedding
        embedding = self.fc(features)
        # embedding = F.normalize(embedding, dim=-1)
        return embedding
    
    def _preprocess_audio(self, audio):
        batch_size = audio.size(0)
        audio_len = audio.size(1)
        if audio_len > self.max_samples:
            audio = audio[:, :self.max_samples]
        elif audio_len < self.max_samples:
            padding = self.max_samples - audio_len
            audio = torch.nn.functional.pad(audio, (0, padding))
        return audio

# Example usage
# audio_encoder = AudioEncoder(output_dim=512, sample_rate=16000)
# audio = torch.randn(4, 240000)  # Batch of 4 audio clips (~15s at 16kHz)
# embedding = audio_encoder(audio)
# print(embedding.shape)  # torch.Size([4, 512])




import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

class AudioEncoder_2(nn.Module):
    def __init__(self, output_dim=512, sample_rate=8000, n_mels=40, max_audio_length=5.0):
        super(AudioEncoder_2, self).__init__()
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)  # 5s at 8kHz = 40,000 samples
        
        # Mel-spectrogram transformation (lightweight)
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,        # Reduced from 1024
            hop_length=128,   # Reduced from 256
            n_mels=n_mels,    # Reduced to 40
            f_min=50.0,
            f_max=4000.0      # Adjusted for 8kHz
        )
        self.to_db = T.AmplitudeToDB(top_db=80.0)
        
        # Simple CNN architecture (~1M parameters)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Small channel count
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling to fixed size
        )
        
        # Fully connected layer
        self.fc = nn.Linear(64, output_dim)  # Simple projection to 512-dim
    
    def forward(self, audio):
        # Preprocess audio
        audio = self._preprocess_audio(audio)
        
        # Compute mel-spectrogram
        spec = self.mel_spec(audio)
        spec = self.to_db(spec)  # Convert to dB scale
        spec = spec.unsqueeze(1)  # Add channel dim: (batch_size, 1, n_mels, time)
        
        # CNN feature extraction
        features = self.cnn(spec)  # (batch_size, 64, 1, 1)
        features = features.view(features.size(0), -1)  # Flatten: (batch_size, 64)
        
        # Project to output_dim
        embedding = self.fc(features)  # (batch_size, 512)
        return embedding
    
    def _preprocess_audio(self, audio):
        batch_size = audio.size(0)
        audio_len = audio.size(1)
        if audio_len > self.max_samples:
            audio = audio[:, :self.max_samples]
        elif audio_len < self.max_samples:
            padding = self.max_samples - audio_len
            audio = torch.nn.functional.pad(audio, (0, padding))
        return audio
    


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

# class AudioEncoder(nn.Module):
#     def __init__(self, output_dim=512, sample_rate=16000, n_mels=64, max_audio_length=15.0):
#         super(AudioEncoder, self).__init__()
#         self.sample_rate = sample_rate
#         self.max_samples = int(max_audio_length * sample_rate)
#         self.n_mels = n_mels

#         # Mel-spectrogram transformation
#         self.mel_spec = T.MelSpectrogram(
#             sample_rate=sample_rate,
#             n_fft=1024,
#             hop_length=256,
#             n_mels=n_mels,
#             f_min=50.0,
#             f_max=14000.0
#         )
#         self.to_db = T.AmplitudeToDB(top_db=80.0)

#         # Lightweight CNN backbone
#         self.cnn = nn.Sequential(
#             # Conv Block 1: Input (batch, 1, n_mels, time)
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 32, n_mels/2, time/2)

#             # Conv Block 2
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 64, n_mels/4, time/4)

#             # Conv Block 3
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 128, n_mels/8, time/8)
#         )

#         # Global average pooling to reduce spatial dimensions
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (batch, 128, 1, 1)

#         # Fully connected layer to project to output_dim
#         self.fc = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, output_dim)
#         )

#     def forward(self, audio):
#         # Preprocess audio to mel-spectrogram
#         audio = self._preprocess_audio(audio)
#         mel = self.mel_spec(audio)  # (batch, n_mels, time)
#         mel = self.to_db(mel)  # Convert to dB scale
#         mel = mel.unsqueeze(1)  # (batch, 1, n_mels, time)

#         # CNN feature extraction
#         features = self.cnn(mel)  # (batch, 128, n_mels/8, time/8)
#         features = self.global_pool(features)  # (batch, 128, 1, 1)
#         features = features.view(features.size(0), -1)  # (batch, 128)

#         # Project to output embedding
#         embedding = self.fc(features)  # (batch, output_dim)
#         # embedding = F.normalize(embedding, dim=-1)  # L2 normalize
#         return embedding

#     def _preprocess_audio(self, audio):
#         batch_size = audio.size(0)
#         audio_len = audio.size(1)
#         if audio_len > self.max_samples:
#             audio = audio[:, :self.max_samples]
#         elif audio_len < self.max_samples:
#             padding = self.max_samples - audio_len
#             audio = torch.nn.functional.pad(audio, (0, padding))
#         return audio

class AudioEncoder(nn.Module):
    def __init__(self, output_dim=512, sample_rate=16000, n_mels=64, max_audio_length=15.0):
        super(AudioEncoder, self).__init__()
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)
        
        # Mel-spectrogram transformation
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            f_min=50.0,
            f_max=14000.0
        )
        self.to_db = T.AmplitudeToDB(top_db=80.0)
        
        # Use a pretrained model from torchaudio.pipelines (e.g., HuBERT_BASE)
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.cnn = bundle.get_model()  # Pretrained HuBERT Base (~94M params, lighter than large variants)
        
        # Freeze pretrained weights
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # HuBERT outputs 768-dim features, so project to 512
        self.fc = nn.Sequential(
            nn.Linear(768, 512),  # Adjust input dim based on model output
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, audio):
        audio = self._preprocess_audio(audio)
        
        # HuBERT expects raw waveforms, not spectrograms
        with torch.no_grad():  # Inference mode for pretrained model
            features, _ = self.cnn(audio)  # (batch_size, seq_len, 768)
            features = features.mean(dim=1)  # Average over time: (batch_size, 768)
        # features, _ = self.cnn(audio)  # (batch_size, seq_len, 768)
        # features = features.mean(dim=1)  
        
        # Project to 512-dim embedding
        embedding = self.fc(features)
        # embedding = F.normalize(embedding, dim=-1)
        return embedding
    
    def _preprocess_audio(self, audio):
        batch_size = audio.size(0)
        audio_len = audio.size(1)
        if audio_len > self.max_samples:
            audio = audio[:, :self.max_samples]
        elif audio_len < self.max_samples:
            padding = self.max_samples - audio_len
            audio = torch.nn.functional.pad(audio, (0, padding))
        return audio

# Example
if __name__ == "__main__":
    audio_encoder = AudioEncoder(output_dim=512)
    audio = torch.randn(4, 240000)  # ~15s at 16kHz
    embedding = audio_encoder(audio)
    print(embedding.shape)  # torch.Size([4, 512])
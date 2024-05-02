import torch
import torch.nn as nn
from torch.nn.functional import interpolate

# The Generator model
class Generator(nn.Module):
    def __init__(self, channels, noise_dim=100, embed_dim=114688, embed_out_dim=256):
        super(Generator, self).__init__()
        self.channels = channels
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        self.embed_out_dim = embed_out_dim

        # Text embedding layers
        # print('hi')
        self.text_embedding = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_out_dim),
            nn.BatchNorm1d(self.embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Generator architecture
        model = []
        # model += self._create_layer(self.noise_dim + self.embed_out_dim, 512, 4, stride=1, padding=0)
        model += self._create_layer(384, 512, 4, stride=1, padding=0)
        model += self._create_layer(512, 256, 4, stride=2, padding=1)
        model += self._create_layer(256, 128, 4, stride=2, padding=1)
        model += self._create_layer(128, 64, 4, stride=2, padding=1)
        model += self._create_layer(64, self.channels, 4, stride=2, padding=1, output=True)

        self.model = nn.Sequential(*model)

    def _create_layer(self, size_in, size_out, kernel_size=4, stride=2, padding=1, output=False):
        layers = [nn.ConvTranspose2d(size_in, size_out, kernel_size, stride=stride, padding=padding, bias=False)]
        if output:
            layers.append(nn.Tanh())  # Tanh activation for the output layer
        else:
            layers += [nn.BatchNorm2d(size_out), nn.ReLU(True)]  # Batch normalization and ReLU for other layers
        return layers

    def forward(self, noise, text):
    # Flatten the text tensor and apply text embedding
        text = text.view(text.shape[0], -1)
        text = self.text_embedding(text)
        print(f'text shape: {text.shape} in Generator')
        print(f'Noise shape: {noise.shape} in Generator')
        z = torch.cat([text, noise], 1)  # Concatenate text embedding with noise
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        print(f'z shape: {z.shape} in Generator')
        return self.model(z)


# The Embedding model
class Embedding(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, x, text):
        print(f'text shape: {text.shape} in Embedding before reshape')
        text = text.reshape(-1, self.embed_dim)  # Reshape the text embeddings
        print(f'text shape: {text.shape} in Embedding')
        print(f'x shape: {x.shape} in Embedding')
        out = self.linear(text)  # Apply the linear transformation
        out = out.reshape(x.size(0), -1, 1, 1)  # Reshape the output to match the input
        out = out.repeat(1, 1, x.size(2), x.size(2))  # Repeat the output to match the input size
        return torch.cat([x, out], 1)  # Concatenate the output with the input


# The Discriminator model
class Discriminator(nn.Module):
    def __init__(self, channels, embed_dim=256, embed_out_dim=128):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.embed_dim = embed_dim
        self.embed_out_dim = embed_out_dim

        # Discriminator architecture
        self.model = nn.Sequential(
            *self._create_layer(self.channels, 64, 4, 2, normalize=False),
            *self._create_layer(64, 128, 4, 2),
            *self._create_layer(128, 256, 4, 2),
            *self._create_layer(256, 256, 4, 2)
        )

        self.text_embedding = Embedding(self.embed_dim, self.embed_out_dim)  # Text embedding module

        # Adjust the number of input channels here
        self.output = nn.Sequential(
            nn.Conv2d(256 + self.embed_out_dim, 1, 4, 1, 0, bias=False), 
            nn.Sigmoid()
        )

    def _create_layer(self, size_in, size_out, kernel_size=4, stride=2, padding=1, normalize=True):
        layers = [nn.Conv2d(size_in, size_out, kernel_size=kernel_size, stride=stride, padding=padding)]
        if normalize:
            layers.append(nn.InstanceNorm2d(size_out, track_running_stats=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, x, text):
        text = text.reshape(x.size(0), -1)
        x_out = self.model(x)  # Extract features from the input using the discriminator architecture
        print("No problem w x")
        out = self.text_embedding(x_out, text)  # Apply text embedding and concatenate with the input features
        print(f'out shape: {out.shape} in Discriminator')
        out = self.output(out)  # Final discriminator output
        return out.squeeze(), x_out
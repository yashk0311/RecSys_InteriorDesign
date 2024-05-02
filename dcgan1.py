import torch
import torch.nn as nn

# The Generator model
class Generator(nn.Module):
    def __init__(self, channels, noise_dim=100, embed_dim=1024, embed_out_dim=128):
        super(Generator, self).__init__()
        self.channels = channels
        self.noise_dim = noise_dim #100
        self.embed_dim = embed_dim #768
        self.embed_out_dim = embed_out_dim #128

        # Text embedding layers
        self.text_embedding = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_out_dim),
            nn.BatchNorm1d(self.embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Generator architecture
        model = []
        # model += self._create_layer(self.noise_dim + self.embed_out_dim, 512, 4, stride=1, padding=0)
        model += self._create_layer(10100, 256, 4, stride=2, padding=1)
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
        # Apply text embedding to the input text
        # print(f'text shape before embedding: {text.shape}')
        text = self.text_embedding(text)
        # print(f'text shape after embedding: {text.shape}')
        text = text.view(text.shape[0], -1, 1, 1)  # Reshape to match the generator input size
        # print(f'text shape after reshaping: {text.shape}')
        noise = noise.view(noise.shape[0], -1, 1, 1)  # Reshape the noise vector
        # print(f'noise shape: {noise.shape}')
        z = torch.cat([text, noise], 1)  # Concatenate text embedding with noise
        # print(f'z shape: {z.shape}')

        return self.model(z)


# The Embedding model
class Embedding(nn.Module):
    def __init__(self, size_in, size_out):
        super(Embedding, self).__init__()
        self.text_embedding = nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.BatchNorm1d(size_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, text):
        embed_out = self.text_embedding(text)
        # print(f'embed_out shape: {embed_out.shape}')
        embed_out = embed_out.view(2, 100, 10, 10) 
        # print(f'embed_out shape after view: {embed_out.shape}')

        # Define the convolutional layer
        conv = nn.Conv2d(embed_out.shape[1], 512, kernel_size=1, stride=1, padding=0)

        # Define the upsampling layer
        upsample = nn.Upsample(size=(28, 28))

        # Pass embed_out through the convolutional layer
        embed_out = conv(embed_out)

        # Pass embed_out through the upsampling layer
        embed_out_resize = upsample(embed_out)
        # print(f'embed_out_resize shape: {embed_out_resize.shape}')
        # print(f'x shape: {x.shape}')
        out = torch.cat([x, embed_out_resize], 1)  # Concatenate text embedding with the input feature map
        return out


# The Discriminator model
class Discriminator(nn.Module):
    def __init__(self, channels, embed_dim=1024, embed_out_dim=128):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.embed_dim = embed_dim #100
        self.embed_out_dim = embed_out_dim #100

        # Discriminator architecture
        self.model = nn.Sequential(
            *self._create_layer(self.channels, 64, 4, 2, 1, normalize=False),
            *self._create_layer(64, 128, 4, 2, 1),
            *self._create_layer(128, 256, 4, 2, 1),
            *self._create_layer(256, 512, 4, 2, 1)
        )
        self.text_embedding = Embedding(self.embed_dim, self.embed_out_dim)  # Text embedding module
        self.output = nn.Sequential(
            nn.Conv2d(924 + self.embed_out_dim, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )

    def _create_layer(self, size_in, size_out, kernel_size=4, stride=2, padding=1, normalize=True):
        layers = [nn.Conv2d(size_in, size_out, kernel_size=kernel_size, stride=stride, padding=padding)]
        if normalize:
            layers.append(nn.BatchNorm2d(size_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, x, text):
        x_out = self.model(x)  # Extract features from the input using the discriminator architecture
        # print(f'x shape in descriminator: {x.shape}')
        # print(f'text shape before embedding in descriminator: {text.shape}')
        # print(f'x_out shape in descriminator: {x_out.shape}')

        out = self.text_embedding(x_out, text)  # Apply text embedding and concatenate with the input features
        # print(f'out shape in descriminator: {out.shape}')
        out = self.output(out)  # Final discriminator output
        # print(f'out shape in descriminator after output: {out.shape}')
        return out.squeeze(), x_out
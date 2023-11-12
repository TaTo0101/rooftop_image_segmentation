### Script for generating a U-net Neural Network
import torch
import torch.nn as nn


# The U-net architecture consist of a CNN encoder and a CNN decoder with the ability for some activations
# to directly skip from encoder to decoder layer
# The kernel sizes and channels mimick the architecture from the paper
class UNet(nn.Module):
    def __init__(self, input_w: int, input_h: int):
        super().__init__()

        ## Encoder
        # Consists of five blocks, where each block consists of two convolutional layers and a max pooling layer,
        # except for the last block. To be able to incoporate skip connections, we have to seperate pooling from the
        # convolution at each encoder layer, and the transposed conv at each decoder layer
        # Since pooling is the same for each encoder layer, we only need to define it once
        # Paper uses cropping later for skip connections, we will use padding in conv pass. Also ensures, that
        # output image is identical to input image
        self.encoder_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 1: Input Size: input_w x input_h x 3
        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # input_w x input_h x 64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # input_w x input_h x 64
            nn.ReLU()
        )
        # After pooling output size: (input_w) / 2 x (input_h) / 2  x 64
        # When input_w = input_h = 256, output size here is 128 x 128 x 64

        # Block 2: Input Size: (input_w) / 2 x (input_h) / 2  x 64
        self.encoder_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # input_w / 2  x input_w / 2 x 128
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # input_w / 2 x input_w / 2 x 128
            nn.ReLU()
        )
        # After pooling output size: (input_w / 2) / 2 x (input_h / 2) / 2 x 128
        # When input_w = input_h = 256, output size here is 64 x 64 x 128

        # Block 3: Input Size: (input_w / 2) / 2 x (input_h / 2) / 2 x 128
        self.encoder_block_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # (input_w / 2) / 2 x (input_h / 2) / 2 x 256
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # (input_w / 2) / 2 x (input_h / 2) / 2 x 256
            nn.ReLU()
        )
        # After pooling output size: ((input_w / 2) / 2) / 2 x ((input_h / 2) / 2) / 2 x 256
        # When input_w = input_h = 256, output size here is 32 x 32 x 256

        # Block 4: Input Size: ((input_w / 2) / 2) / 2 x ((input_h / 2) / 2) / 2 x 256
        self.encoder_block_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # ((input_w / 2) / 2) / 2 x ((input_h / 2) / 2) / 2 x 512
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            #  ((input_w / 2) / 2) / 2 x ((input_h / 2) / 2) / 2 x 512
            nn.ReLU()
        )
        # After pooling output size: (((input_w / 2) / 2) / 2) / 2 x
        # (((input_h / 2) / 2) / 2) / 2 x 512
        # When input_w = input_h = 256, output size here is 16 x 16 x 512

        # Block 5: Input Size: (((input_w / 2) / 2) / 2) / 2 x
        # (((input_h / 2) / 2) / 2) / 2 x 512
        self.encoder_block_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            # (((input_w / 2) / 2) / 2) / 2 x (((input_h / 2) / 2) / 2) / 2 x 1024
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            # (((input_w / 2) / 2) / 2) / 2 x (((input_h / 2) / 2) / 2) / 2 x 1024
            nn.ReLU()
        )
        # When input_w = input_h = 256, output size here is 16 x 16 x 1024

        ## Decoder
        # Input sizes, match output size of encoder block and output sizes match input size of encoder block
        # To be able to incorporate skip connections, first Conv2d pass has original input channel size
        # Here we need to define the transpose pass outside of conv pass, due to varying channel size
        # Furthermore, we either need to crop the feature image from contracting path, pad it, or pad in the upconv pass
        # to concat encoder output and decoder input. Paper uses cropping, we will try using padding in conv layers.
        # Block 1: Input Size: (((input_w / 2) / 2) / 2) / 2 x
        # (((input_h / 2) / 2) / 2) / 2 x 1024
        self.decoder_upconv_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Block 2:
        self.decoder_upconv_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Block 3:
        self.decoder_upconv_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Block 4:
        self.decoder_upconv_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_block_4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Output Layer (channel_out = 1, since we only predict rooftops)
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        ## Encoder
        # Block 1 Conv pass
        x_e1 = self.encoder_block_1(x)
        # Block 1 Pooling
        x_e1p = self.encoder_pooling(x_e1)

        # Block 2 Conv pass
        x_e2 = self.encoder_block_2(x_e1p)
        # Block 2 Pooling
        x_e2p = self.encoder_pooling(x_e2)

        # Block 3 Conv pass
        x_e3 = self.encoder_block_3(x_e2p)
        # Block 3 Pooling
        x_e3p = self.encoder_pooling(x_e3)

        # Block 4 Conv pass
        x_e4 = self.encoder_block_4(x_e3p)
        # Block 4 Pooling
        x_e4p = self.encoder_pooling(x_e4)

        # Block 5 Conv pass
        x_e5 = self.encoder_block_5(x_e4p)

        ## Decoder
        # Upconv Block 1
        x_d1 = self.decoder_upconv_1(x_e5)
        # Conv Pass Block 1 (with skip connection to encoder block 4)
        x_d1 = self.decoder_block_1(torch.cat([x_d1, x_e4], dim=1))

        # Upconv Block 2
        x_d2 = self.decoder_upconv_2(x_d1)
        # Conv Pass Block 2 (with skip connection to encoder block 4)
        x_d2 = self.decoder_block_2(torch.cat([x_d2, x_e3], dim=1))

        # Upconv Block 3
        x_d3 = self.decoder_upconv_3(x_d2)
        # Conv Pass Block 3 (with skip connection to encoder block 4)
        x_d3 = self.decoder_block_3(torch.cat([x_d3, x_e2], dim=1))

        # Upconv Block 4
        x_d4 = self.decoder_upconv_4(x_d3)
        # Conv Pass Block 4 (with skip connection to encoder block 4)
        x_d4 = self.decoder_block_4(torch.cat([x_d4, x_e1], dim=1))

        # Output layer
        y = self.outconv(x_d4)

        return y

# Test
if __name__ == "__main__":
    test_input = torch.randn(4, 3, 256, 256)
    model = UNet(256, 256)

    output = model(test_input)
    print("Output Shape is: ", output.size())


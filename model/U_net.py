import torch
import torch.nn as nn
from torch import optim

class Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_10 = nn.Conv1d(in_ch, in_ch, 10, stride=1, padding='same',groups=in_ch, bias=False)
        self.conv_20 = nn.Conv1d(in_ch, in_ch, 20, stride=1, padding='same',groups=in_ch, bias=False)
        self.conv_40 = nn.Conv1d(in_ch, in_ch, 40, stride=1, padding='same',groups=in_ch, bias=False)
        self.conv_1 = nn.Conv1d(in_ch*3, out_ch, 1, stride=1, padding='same', bias=False)
        
        self.batch_norm_1 = nn.BatchNorm1d(in_ch*3)
        self.batch_norm_2 = nn.BatchNorm1d(out_ch)
        self.relu =  nn.ReLU()

    def forward(self, x):
        x_10 = self.conv_10(x)
        x_20 = self.conv_20(x)
        x_40 = self.conv_40(x)

        x_concat = torch.cat((x_10, x_20,x_40), dim=1)
        x_concat = self.batch_norm_1(x_concat)
        x_concat =  self.relu(x_concat)

        x_out = self.conv_1(x_concat)
        x_out = self.batch_norm_2(x_out)
        x_out =  self.relu(x_out)

        return x_out

class Conv_block_2x(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block1 = Conv_block(in_ch,out_ch)
        self.block2 = Conv_block(out_ch,out_ch)

    def forward(self, x):
        x1 = self.block1(x)
        x_out = self.block2(x1)

        return x_out

class Encoder(nn.Module):
    def __init__(self, chs=(17,32,64,128,256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Conv_block_2x(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool1d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64, 32, 17)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.conv_blocks = nn.ModuleList([Conv_block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.conv_blocks_2x = nn.ModuleList([Conv_block_2x(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 


    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-2):
            _, _, L = encoder_features[i].shape
            up_scale_nn =  nn.Upsample(size=L)

            x        = up_scale_nn(x)
            x        = self.conv_blocks[i](x)
            x        = torch.cat([x, encoder_features[i]], dim=1)
            x        = self.conv_blocks_2x[i](x)
        return x

class UNet(nn.Module):
    def __init__(self, enc_chs=(17,32,64,128,256,512), dec_chs=(512,256, 128, 64, 32, 17),lr=1e-3):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        =  nn.Sequential(
            nn.Conv1d(32,32, 40, stride=1, padding='same',groups=32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1, stride=1, padding='same', bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()


    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        return out

    def predict(self, x):
        with torch.no_grad():
            outputs = self(x)
            outputs = torch.round(outputs)
            
        return outputs








# class YourModel(nn.Module):
#     def __init__(self,resblock):
#         super(YourModel, self).__init__()

#         def conv_block(inp,out_ch,x):
#             conv_10 = nn.Conv1d(inp, inp, 10, stride=1, padding='same',groups=inp, bias=False)
#             conv_20 = nn.Conv1d(inp, inp, 20, stride=1, padding='same',groups=inp, bias=False)
#             conv_40 = nn.Conv1d(inp, inp, 20, stride=1, padding='same',groups=inp, bias=False)
#             conv_1 = nn.Conv1d(inp*3, out, 1, stride=1, padding='same',groups=inp, bias=False)
#             batch_norm = nn.BatchNorm1d(inp*3)
#             relu =  nn.ReLU()

#             x_10 = conv_10(x)
#             x_20 = conv_20(x)
#             x_40 = conv_40(x)

#             x_concat = torch.cat((x_10, x_20,x_40), dim=1)
#             x_concat = batch_norm(x_concat)
#             x_concat =  relu(x_concat)

#             x_out = conv_1(x_concat)
#             x_out = batch_norm(x_out)
#             x_out =  relu(x_out)

#             return x_out

#         def conv_bn(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True)
#             )


#         self.model = nn.Sequential(
#             conv_bn(  3,  32, 2), 
#             resblock( 32,  64, 1),
#             resblock( 64, 128, 2),
#             resblock(128, 128, 1),
#             resblock(128, 256, 2),
#             resblock(256, 256, 1),
#             resblock(256, 512, 2),
#             resblock(512, 512, 1),
#             resblock(512, 512, 1),
#             resblock(512, 512, 1),
#             resblock(512, 512, 1),
#             resblock(512, 512, 1),
#             resblock(512, 1024, 2),
#             resblock(1024, 100, 1),
#             nn.AvgPool2d(7),
#         )
#         self.fc = nn.Sequential(
#             nn.Dropout(0.25),
#             nn.Linear(100, 15),
#             nn.LogSoftmax(dim=1)
#         )

#     def get_optimizer(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr = hp.learning_rate)
#         return optimizer
    
#     def get_loss_fn(self):
#         return nn.NLLLoss()


#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(-1, 100)
#         x = self.fc(x)
#         return x
    
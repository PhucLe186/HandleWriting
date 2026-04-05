import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, vocab_size, img_height=32, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()

        # ── CNN Backbone (Chuẩn 100% không độn thêm bất cứ lớp nào) ──
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),   # 0
            nn.BatchNorm2d(64),                           # 1
            nn.ReLU(True),                                # 2
            nn.MaxPool2d(2, 2),                           # 3

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 4
            nn.BatchNorm2d(128),                          # 5
            nn.ReLU(True),                                # 6
            nn.MaxPool2d(2, 2),                           # 7

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# 8
            nn.BatchNorm2d(256),                          # 9
            nn.ReLU(True),                                # 10
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# 11
            nn.BatchNorm2d(256),                          # 12
            nn.ReLU(True),                                # 13
            nn.MaxPool2d((2, 1)),                         # 14

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),# 15
            nn.BatchNorm2d(512),                          # 16
            nn.ReLU(True),                                # 17
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# 18
            nn.BatchNorm2d(512),                          # 19
            nn.ReLU(True),                                # 20
            nn.MaxPool2d((2, 1)),                         # 21

            # Block 5 
            nn.Conv2d(512, 512, kernel_size=(2, 2)),      # 22
            nn.BatchNorm2d(512),                          # 23
            nn.ReLU(True)                                 # 24
        )

        rnn_input_size = 512

        # ── Bidirectional LSTM ──
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # ── Fully Connected ──
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        features = self.cnn(x)              
        features = features.squeeze(2)      
        features = features.permute(0, 2, 1)

        rnn_out, _ = self.rnn(features)     
        logits = self.fc(rnn_out)           
        logits = logits.permute(1, 0, 2)    

        return logits
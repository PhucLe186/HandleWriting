import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, vocab_size, img_height=32, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()

        # ── CNN Backbone ──────────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H/8, W/4

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H/16, W/4

            # Block 5
            nn.Conv2d(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(),
            # Output: (B, 512, 1, W')
        )

        # ── Map CNN output → RNN input size ──────────────
        # Với img_height=32: sau các pool → height = 1, channels = 512
        rnn_input_size = 512

        # ── Bidirectional LSTM ────────────────────────────
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # ── Fully Connected → vocab ───────────────────────
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        # x: (B, 1, H, W)
        features = self.cnn(x)                      # (B, 512, 1, W')
        b, c, h, w = features.size()

        features = features.squeeze(2)              # (B, 512, W')
        features = features.permute(0, 2, 1)        # (B, W', 512)

        rnn_out, _ = self.rnn(features)             # (B, W', hidden*2)
        logits = self.fc(rnn_out)                   # (B, W', vocab_size)
        logits = logits.permute(1, 0, 2)            # (W', B, vocab_size) — yêu cầu của CTCLoss

        return logits

# Khởi tạo model

import torch
import torch.nn as nn
from CBAM_attn import CBAMSpatialAttentionModule  # Importing your CBAM module


class AccidentPredictorGRU(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256, num_layers=1, bidirectional=False):
        super(AccidentPredictorGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * (2 if bidirectional else 1), 1),
            nn.Sigmoid()
        )

    def forward(self, frame_features):
        """
        Args:
            frame_features: (batch_size, T, N, D) - Aggregated frame features from CBAM

        Returns:
            accident_scores: (batch_size, T) - Accident probability per frame
        """
        gru_out, _ = self.gru(frame_features)
        accident_scores = self.fc(gru_out).squeeze(-1)
        return accident_scores


if __name__ == "__main__":
    # Example usage
    batch_size, T, N, D, d = 2, 50, 19, 4096, 16  # Example dimensions
    F_obj = torch.rand(batch_size, T, N, D)  # Object features
    F_frame = torch.rand(batch_size, T, 1, D)   # Frame features
    F_depth = torch.rand(batch_size, T, N, d)   # Depth features

    # Apply CBAM for spatial attention
    cbam_attention = CBAMSpatialAttentionModule(D, d)
    attention_output = cbam_attention(F_obj, F_frame, F_depth)

    # Initialize GRU
    accident_predictor_gru = AccidentPredictorGRU(D, d)

    # Aggregate object features to frame-level features
    frame_features = attention_output.mean(dim=2)  # (batch_size, T, N, D)

    # Apply GRU for temporal accident anticipation
    accident_scores = accident_predictor_gru(frame_features)  # (batch_size, T)

    print("Accident Scores Shape:", accident_scores.shape)
    print("Accident Scores:", accident_scores)

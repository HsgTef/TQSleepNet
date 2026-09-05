import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Per-sample stochastic depth for residual branches."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    """1-D ConvNeXt block with a DropPath residual branch."""

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
                      if layer_scale_init_value > 0 else None)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)
        return identity + self.drop_path(x)


class MultiScaleTemporalAggregator(nn.Module):
    """MSTA: tokenizes each 30-s epoch into a compact multi-scale temporal-token sequence."""

    def __init__(self, in_channels, feature_dim=192, seq_len=8):
        super().__init__()
        self.seq_len = seq_len

        # Multi-scale convolutional branches (large / medium / small receptive fields).
        self.global_branch = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=65, padding=32, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(seq_len)
        )
        self.mid_branch = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=33, padding=16, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(seq_len)
        )
        self.local_branch = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=17, padding=8, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(seq_len)
        )

        # Channel attention re-weights the raw feature map before pooling.
        reduction = max(1, in_channels // 4)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduction, 1), nn.ReLU(),
            nn.Conv1d(reduction, in_channels, 1), nn.Sigmoid()
        )

        # Fuse channel-weighted features with the three scale branches.
        fusion_dim = in_channels + 32 * 3
        self.projector = nn.Sequential(
            nn.Conv1d(fusion_dim, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim)
        )

    def forward(self, x):
        # Channel-attention weighting, then pool to the token length.
        w = self.channel_attn(x)
        feat_weighted = F.adaptive_avg_pool1d(x * w, self.seq_len)

        feat_global = self.global_branch(x)
        feat_mid = self.mid_branch(x)
        feat_local = self.local_branch(x)

        cat = torch.cat([feat_weighted, feat_global, feat_mid, feat_local], dim=1)
        out = self.projector(cat)  # (B, feature_dim, seq_len)
        return out.permute(0, 2, 1)  # (B, seq_len, feature_dim)


class GatedCrossModalFusion(nn.Module):
    """GCMF: bidirectional token-level cross-modal attention with a shared attention
    module and token-level sigmoid gating."""

    def __init__(self, feature_dim=192):
        super().__init__()
        # The two interaction directions (EEG->EOG and EOG->EEG) share one module.
        self.shared_attn = nn.MultiheadAttention(
            feature_dim, num_heads=4, dropout=0.1, batch_first=True
        )

        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(feature_dim, 2),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim), nn.ReLU(), nn.Dropout(0.1)
        )

    def forward(self, eeg_seq, eog_seq):
        # Token-level cross-modal attention (residual).
        eeg_att, _ = self.shared_attn(eeg_seq, eog_seq, eog_seq)
        eog_att, _ = self.shared_attn(eog_seq, eeg_seq, eeg_seq)
        eeg_enh = eeg_seq + eeg_att
        eog_enh = eog_seq + eog_att

        # Token-level sigmoid gating.
        gates = self.gate_net(torch.cat([eeg_enh, eog_enh], dim=-1))
        w_eeg, w_eog = gates[..., 0:1], gates[..., 1:2]
        fused_seq = w_eeg * eeg_enh + w_eog * eog_enh

        # Token-averaged refined representations for the epoch-level branch / aux heads.
        fused_vec = fused_seq.mean(dim=1)
        eeg_vec_enh = eeg_enh.mean(dim=1)
        eog_vec_enh = eog_enh.mean(dim=1)
        gate_w_vec = gates.mean(dim=1)

        return self.fusion_proj(fused_vec), eeg_vec_enh, eog_vec_enh, gate_w_vec


class StagePrototypeDecoder(nn.Module):
    """SQD: learnable sleep-stage query decoder producing stage logits."""

    def __init__(self, feature_dim=192, num_stages=5):
        super().__init__()
        self.num_stages = num_stages
        self.feature_dim = feature_dim

        # Learnable stage queries (prototypes).
        self.prototypes = nn.Parameter(torch.randn(num_stages, feature_dim) * 0.02)
        nn.init.orthogonal_(self.prototypes)

        self.query_refine = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim), nn.ReLU(), nn.Dropout(0.1)
        )

        # Expand the single fused vector into a pseudo key/value sequence.
        self.feat_expand = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * num_stages),
            nn.ReLU(), nn.Dropout(0.1)
        )

        self.decoder_attn = nn.MultiheadAttention(
            feature_dim, num_heads=4, dropout=0.1, batch_first=True
        )

        self.logit_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, feature_vector):
        B = feature_vector.size(0)
        queries = self.query_refine(self.prototypes).unsqueeze(0).expand(B, -1, -1)
        kv_seq = self.feat_expand(feature_vector).view(B, self.num_stages, self.feature_dim)
        decoded, _ = self.decoder_attn(queries, kv_seq, kv_seq)
        logits = self.logit_proj(decoded).squeeze(-1)  # (B, num_stages)
        return logits


class TQSleepNet(nn.Module):
    """TQSleepNet: Multi-Scale Temporal-Token Query Network for sleep staging.

    Pipeline:
        raw EEG/EOG -> ConvNeXt encoder -> MSTA (multi-scale temporal tokens)
        -> GCMF (shared cross-modal attention + gating) -> SQD (stage-query decoder)
        -> Bi-LSTM inter-epoch context -> final classification.
    """

    def __init__(self, eeg_channels=1, eog_channels=1, num_classes=5,
                 context=30, has_eog=True, token_len=8):
        super().__init__()
        self.context = context          # context window length (informational)
        self.has_eog = has_eog
        self.token_len = token_len      # M: temporal tokens per epoch
        query_dim = 192
        seq_dim = 128

        # ---- Stage 1: feature extraction ----
        self.eeg_encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(eeg_channels, eeg_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(eeg_channels), nn.ReLU(),
                nn.Conv1d(eeg_channels, 48, 15, padding=7, bias=False),
                nn.BatchNorm1d(48), nn.ReLU(), nn.MaxPool1d(2, 2)
            ),
            nn.Sequential(*[ConvNeXtBlock(48, drop_path=0.08 * i / 6) for i in range(6)])
        )
        self.eeg_expand = nn.Sequential(
            nn.Conv1d(48, 96, 3, padding=1, bias=False),
            nn.BatchNorm1d(96), nn.ReLU(), nn.AdaptiveAvgPool1d(128)
        )

        if has_eog:
            self.eog_encoder = nn.Sequential(
                nn.Sequential(
                    nn.Conv1d(eog_channels, eog_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(eog_channels), nn.ReLU(),
                    nn.Conv1d(eog_channels, 24, 15, padding=7, bias=False),
                    nn.BatchNorm1d(24), nn.ReLU(), nn.MaxPool1d(2, 2)
                ),
                nn.Sequential(*[ConvNeXtBlock(24, drop_path=0.08 * i / 4) for i in range(4)])
            )
            self.eog_expand = nn.Sequential(
                nn.Conv1d(24, 48, 3, padding=1, bias=False),
                nn.BatchNorm1d(48), nn.ReLU(), nn.AdaptiveAvgPool1d(128)
            )

        # ---- Stage 2: multi-scale temporal aggregation (MSTA) ----
        self.eeg_aggregator = MultiScaleTemporalAggregator(96, query_dim, token_len)
        if has_eog:
            self.eog_aggregator = MultiScaleTemporalAggregator(48, query_dim, token_len)

        # ---- Stage 3: gated cross-modal fusion (GCMF) + stage-query decoder (SQD) ----
        self.modal_fusion = GatedCrossModalFusion(query_dim)
        self.stage_decoder = StagePrototypeDecoder(query_dim, num_classes)

        self.aux_eeg_head = nn.Linear(query_dim, num_classes)
        self.aux_eog_head = nn.Linear(query_dim, num_classes)

        # ---- Stage 4: inter-epoch context modeling (Bi-LSTM) ----
        self.compressor = nn.Sequential(
            nn.Linear(query_dim, seq_dim),
            nn.LayerNorm(seq_dim), nn.ReLU()
        )
        self.bi_lstm = nn.LSTM(
            seq_dim, seq_dim, batch_first=True, bidirectional=True,
            dropout=0.15, num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(seq_dim * 2, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, eeg_x, eog_x=None):
        """Args:
            eeg_x: (B, N, C_eeg, T) raw EEG window sequence.
            eog_x: (B, N, C_eog, T) raw EOG window sequence (optional).
        Returns:
            (main_logits, stage_logits, stage_prob, aux_eeg, aux_eog, gate_w)
        """
        B, N, C_eeg, T = eeg_x.shape

        # 1. Feature extraction
        eeg = self.eeg_encoder(eeg_x.view(B * N, C_eeg, T))
        eeg = self.eeg_expand(eeg)

        if self.has_eog and eog_x is not None:
            eog = self.eog_encoder(eog_x.view(B * N, eog_x.size(2), T))
            eog = self.eog_expand(eog)
        else:
            eog = torch.zeros(B * N, 48, 128, device=eeg.device)

        # 2. Aggregation
        eeg_vec = self.eeg_aggregator(eeg)
        eog_vec = self.eog_aggregator(eog) if self.has_eog else torch.zeros_like(eeg_vec)

        # 3. Fusion & stage decoding
        fused_vec, eeg_enh, eog_enh, gate_w = self.modal_fusion(eeg_vec, eog_vec)
        stage_logits = self.stage_decoder(fused_vec)

        aux_eeg = self.aux_eeg_head(eeg_enh)
        aux_eog = self.aux_eog_head(eog_enh)

        stage_logits = stage_logits.view(B, N, -1)
        aux_eeg = aux_eeg.view(B, N, -1)
        aux_eog = aux_eog.view(B, N, -1)
        gate_w = gate_w.view(B, N, -1)

        # 4. Context modeling
        seq_feat = self.compressor(fused_vec).view(B, N, -1)
        lstm_out, _ = self.bi_lstm(seq_feat)
        main_logits = self.classifier(lstm_out)

        return main_logits, stage_logits, F.softmax(stage_logits, -1), aux_eeg, aux_eog, gate_w


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing cross-entropy loss."""

    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TQSleepNetLoss(nn.Module):
    """Total loss = 1.0 * main + 0.5 * stage + 0.05 * modality-aux."""

    def __init__(self, main_weight=1.0, stage_weight=0.5, modal_weight=0.05):
        super().__init__()
        self.main_weight = main_weight
        self.stage_weight = stage_weight
        self.modal_weight = modal_weight
        self.ce_loss = LabelSmoothingLoss(classes=5, smoothing=0.1)

    def forward(self, main_output, stage_output, targets,
                eeg_logits=None, eog_logits=None, mask=None):
        if main_output.dim() == 3:
            B, N, C = main_output.shape
            main_output = main_output.reshape(-1, C)
            stage_output = stage_output.reshape(-1, C)
            targets = targets.reshape(-1)
            if eeg_logits is not None:
                eeg_logits = eeg_logits.reshape(-1, C)
            if eog_logits is not None:
                eog_logits = eog_logits.reshape(-1, C)
            if mask is not None:
                mask = mask.reshape(-1)
                main_output = main_output[mask]
                stage_output = stage_output[mask]
                targets = targets[mask]
                if eeg_logits is not None:
                    eeg_logits = eeg_logits[mask]
                if eog_logits is not None:
                    eog_logits = eog_logits[mask]

        main_loss = self.ce_loss(main_output, targets)
        stage_loss = self.ce_loss(stage_output, targets)
        total_loss = self.main_weight * main_loss + self.stage_weight * stage_loss

        modal_loss = torch.tensor(0.0, device=main_output.device)
        if eeg_logits is not None and eog_logits is not None:
            eeg_loss = self.ce_loss(eeg_logits, targets)
            eog_loss = self.ce_loss(eog_logits, targets)
            modal_loss = (eeg_loss + eog_loss) / 2
            total_loss += self.modal_weight * modal_loss

        return total_loss, main_loss, stage_loss, modal_loss


if __name__ == "__main__":
    # EEG+EOG configuration (e.g., SleepEDF: 1 EEG + 1 EOG, context=30, M=8).
    model = TQSleepNet(eeg_channels=1, eog_channels=1, num_classes=5,
                       context=30, has_eog=True, token_len=8)
    eeg = torch.randn(2, 30, 1, 3000)
    eog = torch.randn(2, 30, 1, 3000)
    main, stage, stage_prob, aux_eeg, aux_eog, gate_w = model(eeg, eog)
    print("main:", tuple(main.shape))
    print("stage:", tuple(stage.shape))
    print("gate:", tuple(gate_w.shape))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f} M")

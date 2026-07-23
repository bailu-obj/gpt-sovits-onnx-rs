import sys
sys.path.append('./')
import torch
import torchaudio
from torch import nn
from feature_extractor import cnhubert
from text import cleaned_text_to_sequence
import soundfile
import os
import json
from transformers import AutoModelForMaskedLM, AutoTokenizer
from module import commons
from module.models_onnx import SynthesizerTrn, symbols_v1, symbols_v2
from AR.models.t2s_lightning_module_onnx import Text2SemanticLightningModule
import argparse
from torch import Tensor
import torch.nn.functional as F

from AR.models.t2s_model_onnx import sample
from sv import SV
import kaldi as Kaldi
from process_ckpt import get_sovits_version_from_path_fast

V2PRO_SET = {"v2Pro", "v2ProPlus"}

# Optional batched T2S ONNX export (off by default).
# Set GSV_EXPORT_T2S_BATCH=1 to add dynamic batch axis 0 on stage-decoder I/O.
# Requires a matching batched Rust ORT path — not enabled in gpt-sovits-onnx-rs yet.
_EXPORT_T2S_BATCH = os.environ.get("GSV_EXPORT_T2S_BATCH", "0") == "1"
# Export stage-decoder K/V as single-row deltas (default on). Set GSV_EXPORT_KV_DELTA=0
# to emit full caches for older runtimes.
_EXPORT_KV_DELTA = os.environ.get("GSV_EXPORT_KV_DELTA", "1") == "1"
# FS decoder BERT layout: native [B,T,1024] by default. Set GSV_EXPORT_BERT_BFT=1 for
# legacy [B,1024,T] (requires transpose inside the graph).
_EXPORT_BERT_BFT = os.environ.get("GSV_EXPORT_BERT_BFT", "0") == "1"

# PyTorch 2.6+ defaults to dynamo/torch.export ONNX; GPT-SoVITS needs legacy export.
_LEGACY_ONNX_KWARGS = {"dynamo": False}


def is_v2pro(version: str) -> bool:
    return version in V2PRO_SET


def resolve_version(vits_path: str, version: str, auto_version: bool) -> str:
    if not auto_version:
        return version
    detected = get_sovits_version_from_path_fast(vits_path)
    print(f"Auto-detected SoVITS version: {detected}")
    return detected


sv_cn_model = None


def init_sv_cn(device, is_half):
    global sv_cn_model
    sv_cn_model = SV(device, is_half)

EOS = 1024

def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    hann_window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

class T2SEncoder(nn.Module):
    def __init__(self, t2s, vits):
        super().__init__()
        self.vits = vits
    
    def forward(self, ssl_content):
        codes = self.vits.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        x = prompt_semantic.unsqueeze(0)
        # x -> all_phoneme_ids
        # all_phoneme_ids.len
        return x

class T2SModel(nn.Module):
    def __init__(self, t2s_path, vits_model):
        super().__init__()
        dict_s1 = torch.load(t2s_path, map_location="cpu", weights_only=False)
        self.config = dict_s1["config"]
        self.t2s_model = Text2SemanticLightningModule(self.config, "ojbk", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        self.t2s_model.eval()
        self.vits_model = vits_model.vq_model
        self.hz = 50
        self.max_sec = self.config["data"]["max_sec"]
        # self.t2s_model.model.top_k = torch.LongTensor([self.config["inference"]["top_k"]])
        # self.t2s_model.model.early_stop_num = torch.LongTensor([self.hz * self.max_sec])
        self.t2s_model = self.t2s_model.model
        self.t2s_model.init_onnx()
        self.onnx_encoder = T2SEncoder(self.t2s_model, self.vits_model)
        self.first_stage_decoder = self.t2s_model.first_stage_decoder
        self.stage_decoder = self.t2s_model.stage_decoder

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = torch.LongTensor([self.hz * self.max_sec])
        prompts = self.onnx_encoder(ssl_content)
        if _EXPORT_BERT_BFT:
            bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1).unsqueeze(0)
        else:
            bert = torch.cat([ref_bert, text_bert], 0).unsqueeze(0)
        x = torch.cat([ref_seq, text_seq], 1)
        y_len = prompts.shape[1]
        prefix_len = prompts.shape[1]
        logits, k_cache, v_cache = self.first_stage_decoder(x, prompts, bert)
        y = prompts
        samples = sample(logits, prompts,top_k=15, top_p = 1.0, temperature=1.0)[0].unsqueeze(0)

        stop = False
        for idx in range(0, 1500):
            logits, k_cache, v_cache = self.stage_decoder(y, k_cache, v_cache, y_len, idx)
            samples = sample(logits, y,top_k=15, top_p = 1.0, temperature=1.0)[0].unsqueeze(0)
            y = torch.concat([y, samples], dim=1)
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if y[0, -1] == EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0
        return y[:, prefix_len:].unsqueeze(0)

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name):
        torch.onnx.export(
            self.onnx_encoder,
            (ssl_content),
            f"onnx/{project_name}/{project_name}_t2s_encoder.onnx",
            input_names=["ssl_content"],
            output_names=["prompts"],
            dynamic_axes={
                "ssl_content": {2: "ssl_length"},
            },
            opset_version=20,
            **_LEGACY_ONNX_KWARGS,
        )
        
        prompts = self.onnx_encoder(ssl_content)

        if _EXPORT_BERT_BFT:
            # Legacy [B, 1024, T] — FS module temporarily uses transpose path.
            bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1).unsqueeze(0)
            print("GSV_EXPORT_BERT_BFT=1: FS decoder bert layout [B,1024,T] (legacy)")
            fs_bert_axis = {2: "bert_length"}
        else:
            bert = torch.cat([ref_bert, text_bert], 0).unsqueeze(0)
            print("FS decoder bert layout [B,T,1024] (native, no graph Transpose)")
            fs_bert_axis = {1: "bert_length"}
        x = torch.cat([ref_seq, text_seq], 1)
        y_len = prompts.shape[1]

        num_layers = self.t2s_model.num_layers
        # Seq-major [B, T, H*D] — contiguous time-prefix views for ORT (best measured CPU path).
        hidden = self.t2s_model.model_dim
        k_cache = [
            torch.zeros((1, 0, hidden), dtype=x.dtype, device=x.device)
            for _ in range(num_layers)
        ]
        v_cache = [
            torch.zeros((1, 0, hidden), dtype=x.dtype, device=x.device)
            for _ in range(num_layers)
        ]

        fs_dynamic_axes = {
            "x": {1: "x_length"},
            "prompts": {1: "prompts_length"},
            "bert": fs_bert_axis,
        }
        if _EXPORT_T2S_BATCH:
            fs_dynamic_axes["x"][0] = "batch"
            fs_dynamic_axes["bert"][0] = "batch"
            print("GSV_EXPORT_T2S_BATCH=1: exporting FS decoder with dynamic batch axis 0")

        if _EXPORT_BERT_BFT:
            # Temporarily restore transpose inside FS for legacy A/B exports.
            _fs = self.first_stage_decoder
            _orig_forward = _fs.forward

            def _legacy_forward(x_in, prompt, bert_feature, _mod=_fs):
                x_emb = _mod.ar_text_embedding(x_in)
                x_emb = x_emb + _mod.bert_proj(bert_feature.transpose(1, 2))
                x_emb = _mod.ar_text_position(x_emb)
                y = prompt
                x_len = x_emb.shape[1]
                y_emb = _mod.ar_audio_embedding(y)
                y_pos = _mod.ar_audio_position(y_emb)
                xy_pos = torch.concat([x_emb, y_pos], dim=1)
                y_len_local = y_emb.shape[1]
                x_attn_mask_pad = F.pad(
                    torch.zeros((x_len, x_len), dtype=torch.bool),
                    (0, y_len_local),
                    value=True,
                )
                y_attn_mask = F.pad(
                    torch.triu(
                        torch.ones(y_len_local, y_len_local, dtype=torch.bool), diagonal=1
                    ),
                    (x_len, 0),
                    value=False,
                )
                src_len = x_len + y_len_local
                xy_attn_mask = (
                    torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
                    .unsqueeze(0)
                    .expand(_mod.num_head, -1, -1)
                    .view(1, _mod.num_head, src_len, src_len)
                )
                xy_dec, k_c, v_c = _mod.h(
                    xy_pos, mask=xy_attn_mask, k_cache=None, v_cache=None, first_infer=True
                )
                logits_out = _mod.ar_predict_layer(xy_dec[:, -1])
                return logits_out[0], k_c, v_c

            _fs.forward = _legacy_forward  # type: ignore[method-assign]

        torch.onnx.export(
            self.first_stage_decoder,
            (x, prompts, bert),
            f"onnx/{project_name}/{project_name}_t2s_fs_decoder.onnx",
            input_names=["x", "prompts", "bert"],
            output_names=["logits"] + [f"k_cache_{i}" for i in range(num_layers)] + 
                         [f"v_cache_{i}" for i in range(num_layers)],
            dynamic_axes=fs_dynamic_axes,
            verbose=False,
            opset_version=20,
            **_LEGACY_ONNX_KWARGS,
        )
        logits, k_cache, v_cache  = self.first_stage_decoder(x, prompts, bert)
        if _EXPORT_BERT_BFT:
            self.first_stage_decoder.forward = _orig_forward  # type: ignore[method-assign]

        samples = sample(logits, prompts,top_k=15, top_p = 1.0, temperature=1.0)[0].unsqueeze(0)
        y = torch.concat([prompts, samples], dim=1)
        idx = 0
        s_decoder_dynamic_axes = {
            "iy": {1: "iy_length"},
            **{f"ik_cache_{i}": {1: "kv_length"} for i in range(num_layers)},
            **{f"iv_cache_{i}": {1: "kv_length"} for i in range(num_layers)},
        }
        if _EXPORT_T2S_BATCH:
            s_decoder_dynamic_axes["iy"][0] = "batch"
            for i in range(num_layers):
                s_decoder_dynamic_axes[f"ik_cache_{i}"][0] = "batch"
                s_decoder_dynamic_axes[f"iv_cache_{i}"][0] = "batch"
            print("GSV_EXPORT_T2S_BATCH=1: exporting stage decoder with dynamic batch axis 0")

        from AR.models.t2s_model_onnx import T2SStageDecoderDeltaKV

        if _EXPORT_KV_DELTA:
            s_export_mod = T2SStageDecoderDeltaKV(self.stage_decoder)
            print("GSV_EXPORT_KV_DELTA=1: stage decoder emits single-row K/V deltas")
        else:
            s_export_mod = self.stage_decoder
            print("GSV_EXPORT_KV_DELTA=0: stage decoder emits full K/V caches")

        torch.onnx.export(
            s_export_mod,
            (y, k_cache, v_cache, y_len, idx),
            f"onnx/{project_name}/{project_name}_t2s_s_decoder.onnx",
            input_names=["iy"] + [f"ik_cache_{i}" for i in range(num_layers)] + 
                        [f"iv_cache_{i}" for i in range(num_layers)] + ["y_len", "idx"],
            output_names=["logits"] + [f"k_cache_{i}" for i in range(num_layers)] + 
                         [f"v_cache_{i}" for i in range(num_layers)],
            dynamic_axes=s_decoder_dynamic_axes,
            verbose=False,
            opset_version=20,
            **_LEGACY_ONNX_KWARGS,
        )
        

class VitsModel(nn.Module):
    def __init__(self, vits_path, version="v2"):
        super().__init__()
        dict_s2 = torch.load(vits_path, map_location="cpu", weights_only=False)
        self.hps = dict_s2["config"]
        if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            self.hps["model"]["version"] = "v1"
        else:
            self.hps["model"]["version"] = version
        
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        self.vq_model.dec.remove_weight_norm()
        self.version = version
        
    def forward(self, text_seq, pred_semantic, ref_audio, sv_emb=None):
        refer = spectrogram_torch(
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False
        )
        return self.vq_model(pred_semantic, text_seq, refer, sv_emb=sv_emb)[0]


class VitsRefEncoder(nn.Module):
    """Reference path only: ref_audio (+ optional sv_emb) -> ge style tensor."""

    def __init__(self, vits: VitsModel):
        super().__init__()
        self.vq_model = vits.vq_model
        self.hps = vits.hps
        self.version = vits.version

    def forward(self, ref_audio, sv_emb=None):
        refer = spectrogram_torch(
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, refer.size(2)), 1
        ).to(refer.dtype)
        if self.vq_model.version == "v1":
            ge = self.vq_model.ref_enc(refer * refer_mask, refer_mask)
        else:
            ge = self.vq_model.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        if self.vq_model.is_v2pro:
            sv_emb = self.vq_model.sv_emb(sv_emb)
            ge = ge + sv_emb.unsqueeze(-1)
            ge = self.vq_model.prelu(ge)
        return ge


class VitsDecode(nn.Module):
    """Decode path with cached ge: text + pred_semantic + controls -> waveform."""

    def __init__(self, vits: VitsModel):
        super().__init__()
        self.vq_model = vits.vq_model

    def forward(self, text_seq, pred_semantic, ge, noise_scale, speed):
        quantized = self.vq_model.quantizer.decode(pred_semantic)
        if self.vq_model.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, scale_factor=2.0, mode="nearest")

        y_lengths = torch.LongTensor([quantized.size(2)]).to(pred_semantic.device)
        text_lengths = torch.LongTensor([text_seq.size(1)]).to(pred_semantic.device)
        noise_scale = noise_scale[0]
        speed_scalar = speed[0]

        if self.vq_model.is_v2pro:
            ge_ = self.vq_model.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
            x, m_p, logs_p, y_mask = self.vq_model.enc_p(
                quantized, y_lengths, text_seq, text_lengths, ge_, speed_scalar
            )
        else:
            x, m_p, logs_p, y_mask = self.vq_model.enc_p(
                quantized, y_lengths, text_seq, text_lengths, ge, speed_scalar
            )

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.vq_model.flow(z_p, y_mask, g=ge, reverse=True)
        o = self.vq_model.dec((z * y_mask)[:, :, :], g=ge)
        # Keep `speed` as an explicit graph input for runtime control pipelines.
        return o[:, 0, :] + speed.reshape(1, 1) * 1e-7


def export_split_vits(vits, text_seq, pred_semantic, ref_audio, project_name, sv_emb=None):
    """Export optional split VITS graphs (ref + decode). Off by default; see --split-vits-ref."""
    ref_encoder = VitsRefEncoder(vits)
    decode = VitsDecode(vits)
    is_pro = is_v2pro(vits.version)

    noise_scale = torch.tensor([0.5], dtype=torch.float32, device=pred_semantic.device)
    speed = torch.tensor([1.0], dtype=torch.float32, device=pred_semantic.device)

    if is_pro:
        torch.onnx.export(
            ref_encoder,
            (ref_audio, sv_emb),
            f"onnx/{project_name}/{project_name}_vits_ref.onnx",
            input_names=["ref_audio", "sv_emb"],
            output_names=["ge"],
            dynamic_axes={
                "ref_audio": {1: "audio_length"},
            },
            opset_version=20,
            verbose=False,
            **_LEGACY_ONNX_KWARGS,
        )
        ge = ref_encoder(ref_audio, sv_emb)
        torch.onnx.export(
            decode,
            (text_seq, pred_semantic, ge, noise_scale, speed),
            f"onnx/{project_name}/{project_name}_vits_decode.onnx",
            input_names=["text_seq", "pred_semantic", "ge", "noise_scale", "speed"],
            output_names=["audio"],
            dynamic_axes={
                "text_seq": {1: "text_length"},
                "pred_semantic": {2: "pred_length"},
            },
            opset_version=20,
            verbose=False,
            **_LEGACY_ONNX_KWARGS,
        )
    else:
        torch.onnx.export(
            ref_encoder,
            (ref_audio,),
            f"onnx/{project_name}/{project_name}_vits_ref.onnx",
            input_names=["ref_audio"],
            output_names=["ge"],
            dynamic_axes={
                "ref_audio": {1: "audio_length"},
            },
            opset_version=20,
            verbose=False,
            **_LEGACY_ONNX_KWARGS,
        )
        ge = ref_encoder(ref_audio)
        torch.onnx.export(
            decode,
            (text_seq, pred_semantic, ge, noise_scale, speed),
            f"onnx/{project_name}/{project_name}_vits_decode.onnx",
            input_names=["text_seq", "pred_semantic", "ge", "noise_scale", "speed"],
            output_names=["audio"],
            dynamic_axes={
                "text_seq": {1: "text_length"},
                "pred_semantic": {2: "pred_length"},
            },
            opset_version=20,
            verbose=False,
            **_LEGACY_ONNX_KWARGS,
        )
    print(f"#### exported split VITS: {project_name}_vits_ref.onnx + {project_name}_vits_decode.onnx ####")

class GptSoVits(nn.Module):
    def __init__(self, vits, t2s, sv_model=None, version="v2"):
        super().__init__()
        self.vits = vits
        self.t2s = t2s
        self.sv_model = sv_model
        self.version = version
    
    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content):
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        if is_v2pro(self.version):
            audio_16k = torchaudio.functional.resample(ref_audio, self.vits.hps.data.sampling_rate, 16000).float()
            audio_feature = Kaldi.fbank(audio_16k, num_mel_bins=80, sample_frequency=16000, dither=0)
            sv_emb = self.sv_model(audio_feature)
            return self.vits(text_seq, pred_semantic, ref_audio, sv_emb=sv_emb)
        else:
            return self.vits(text_seq, pred_semantic, ref_audio)

    def export(
        self,
        ref_seq,
        text_seq,
        ref_bert,
        text_bert,
        ref_audio,
        ssl_content,
        project_name,
        split_vits_ref=False,
    ):
        self.t2s.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name)
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        if is_v2pro(self.version):
            dummy_audio_16k = torchaudio.functional.resample(ref_audio, self.vits.hps.data.sampling_rate, 16000).float()
            audio_feature = Kaldi.fbank(dummy_audio_16k, num_mel_bins=80, sample_frequency=16000, dither=0)
            print("Exporting SV model...")
            print(audio_feature.shape)
            sv_emb = self.sv_model(audio_feature)
            torch.onnx.export(
                self.sv_model,
                audio_feature,
                f"onnx/{project_name}/sv.onnx",
                input_names=["audio_feature"],
                output_names=["sv_emb"],
                dynamic_axes={
                    "audio_feature": {0: "length"},
                },
                opset_version=20,
                verbose=False,
                **_LEGACY_ONNX_KWARGS,
            )
            torch.onnx.export(
                self.vits,
                (text_seq, pred_semantic, ref_audio, sv_emb),
                f"onnx/{project_name}/{project_name}_vits.onnx",
                input_names=["text_seq", "pred_semantic", "ref_audio", "sv_emb"],
                output_names=["audio"],
                dynamic_axes={
                    "text_seq": {1: "text_length"},
                    "pred_semantic": {2: "pred_length"},
                    "ref_audio": {1: "audio_length"},
                },
                opset_version=20,
                verbose=False,
                **_LEGACY_ONNX_KWARGS,
            )
        else:
            torch.onnx.export(
                self.vits,
                (text_seq, pred_semantic, ref_audio),
                f"onnx/{project_name}/{project_name}_vits.onnx",
                input_names=["text_seq", "pred_semantic", "ref_audio"],
                output_names=["audio"],
                dynamic_axes={
                    "text_seq": {1: "text_length"},
                    "pred_semantic": {2: "pred_length"},
                    "ref_audio": {1: "audio_length"},
                },
                opset_version=20,
                verbose=False,
                **_LEGACY_ONNX_KWARGS,
            )

        if split_vits_ref:
            export_split_vits(
                self.vits,
                text_seq,
                pred_semantic,
                ref_audio,
                project_name,
                sv_emb=sv_emb if is_v2pro(self.version) else None,
            )

class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.ssl = cnhubert.get_model().model

    def forward(self, ref_audio_16k):
        return self.ssl(ref_audio_16k)["last_hidden_state"].transpose(1, 2)
    


class ExportERes2NetV2(nn.Module): # SV model
    def __init__(self, sv_cn_model: SV):
        super(ExportERes2NetV2, self).__init__()
        self.bn1 = sv_cn_model.embedding_model.bn1
        self.conv1 = sv_cn_model.embedding_model.conv1
        self.layer1 = sv_cn_model.embedding_model.layer1
        self.layer2 = sv_cn_model.embedding_model.layer2
        self.layer3 = sv_cn_model.embedding_model.layer3
        self.layer4 = sv_cn_model.embedding_model.layer4
        self.layer3_ds = sv_cn_model.embedding_model.layer3_ds
        self.fuse34 = sv_cn_model.embedding_model.fuse34

    # audio_16k.shape: [1,N]
    def forward(self, audio_16k):
        # 这个 fbank 函数有一个 cache, 不过不要紧，它跟 audio_16k 的长度无关
        # 只跟 device 和 dtype 有关
        x = torch.stack([audio_16k])

        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out3_ds = self.layer3_ds(out3)
        fuse_out34 = self.fuse34(out4, out3_ds)
        return fuse_out34.flatten(start_dim=1, end_dim=2).mean(-1)




class MyBertModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(MyBertModel, self).__init__()
        self.bert = bert_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        res = outputs["hidden_states"][-3][0][1:-1]
        # Phone-level expansion stays in Rust (avoids export subgraph + bugs).
        return res


def export_bert(project_name):
    bert_path = os.environ.get(
        "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    )
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    text = "叹息声一声接着一声传出,木兰对着房门织布.听不见织布机织布的声音,只听见木兰在叹息.问木兰在想什么?问木兰在惦记什么?木兰答道,我也没有在想什么,也没有在惦记什么."
    ref_bert_inputs = tokenizer(text, return_tensors="pt")
    word2ph = []
    for c in text:
        if c in ["，", "。", "：", "？", ",", ".", "?"]:
            word2ph.append(1)
        else:
            word2ph.append(2)
    ref_bert_inputs["word2ph"] = torch.Tensor(word2ph).int()

    bert_model = AutoModelForMaskedLM.from_pretrained(
        bert_path, output_hidden_states=True,
    )
    my_bert_model = MyBertModel(bert_model)

    torch.onnx.export(
        my_bert_model,
        (
            ref_bert_inputs["input_ids"],
            ref_bert_inputs["attention_mask"],
            ref_bert_inputs["token_type_ids"],
        ),
        f"onnx/{project_name}/bert.onnx",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["bert_feature"],
        dynamic_axes={
            "input_ids": {1: "input_ids_length"},
            "attention_mask": {1: "attention_mask_len"},
            "token_type_ids": {1: "token_type_ids_len"},
        },
        opset_version=20,
        verbose=False,
        **_LEGACY_ONNX_KWARGS,
    )
    print("#### exported bert ####")


def export(vits_path, gpt_path, project_name, vits_model="v2", split_vits_ref=False):
    vits = VitsModel(vits_path, version=vits_model)
    gpt = T2SModel(gpt_path, vits)
    sv_model = None
    if is_v2pro(vits_model):
        init_sv_cn("cpu", False)
        sv_model = ExportERes2NetV2(sv_cn_model)
    gpt_sovits = GptSoVits(vits, gpt, sv_model=sv_model, version=vits_model)
    ssl = SSLModel()
    ref_seq = torch.LongTensor([cleaned_text_to_sequence(["n", "i2", "h", "ao3", "a1", ",", "w", "o3", "sh", "i4", "zh", "i4", "n", "eng2", "y", "u3", "y", "in1", "zh", "u4", "sh", "ou3"], version=vits_model)])
    text_seq = torch.LongTensor([cleaned_text_to_sequence(["w", "o3", "sh", "i4", "b", "ai2", "y", "e4", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4"], version=vits_model)])
    ref_bert = torch.zeros((ref_seq.shape[1], 1024)).float()
    text_bert = torch.zeros((text_seq.shape[1], 1024)).float()
    ref_audio = torch.randn((1, 48000 * 5)).float()
    ref_audio_16k = torchaudio.functional.resample(ref_audio, 48000, 16000).float()
    ref_audio_sr = torchaudio.functional.resample(ref_audio, 48000, vits.hps.data.sampling_rate).float()

    try:
        os.mkdir(f"onnx/{project_name}")
    except:
        pass

    ssl_content = ssl(ref_audio_16k).float()

    torch.onnx.export(
        ssl,
        ref_audio_16k,
        f"onnx/{project_name}/ssl.onnx",
        input_names=["ref_audio_16k"],
        output_names=["ssl_content"],
        dynamic_axes={
            "ref_audio_16k": {1: "audio_length"},
        },
        opset_version=20,
        verbose=False,
        **_LEGACY_ONNX_KWARGS,
    )
    export_bert(project_name)
    gpt_sovits.export(
        ref_seq,
        text_seq,
        ref_bert,
        text_bert,
        ref_audio_sr,
        ssl_content,
        project_name,
        split_vits_ref=split_vits_ref,
    )

    a = gpt_sovits(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content).detach().cpu().numpy()
    soundfile.write("out.wav", a, vits.hps.data.sampling_rate)

    if vits_model == "v1":
        symbols = symbols_v1
    else:
        symbols = symbols_v2

    MoeVSConf = {
        "Folder": f"{project_name}",
        "Name": f"{project_name}",
        "Type": "GPT-SoVits",
        "Rate": vits.hps.data.sampling_rate,
        "NumLayers": gpt.t2s_model.num_layers,
        "EmbeddingDim": gpt.t2s_model.embedding_dim,
        "Dict": "BasicDict",
        "BertPath": "chinese-roberta-wwm-ext-large",
        "AddBlank": False,
        "Version": vits_model,
        "IsV2Pro": is_v2pro(vits_model),
    }

    with open(f"onnx/{project_name}.json", 'w') as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--export_name", type=str, required=True, help="Project Name for the exported model")
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        help="vits model version: v2, v2Pro, or v2ProPlus",
    )
    parser.add_argument(
        "--auto-version",
        action="store_true",
        help="detect version from sovits.pth via process_ckpt",
    )
    parser.add_argument(
        "--split-vits-ref",
        action="store_true",
        help="Also export optional {name}_vits_ref.onnx + {name}_vits_decode.onnx (experimental; monolithic still exported)",
    )
    args = parser.parse_args()

    try:
        os.mkdir("onnx")
    except:
        pass
    gpt_path = os.path.join(args.model_path, "gpt.ckpt")
    vits_path = os.path.join(args.model_path, "sovits.pth")
    version = resolve_version(vits_path, args.version, args.auto_version)
    with torch.no_grad():
        export(vits_path, gpt_path, args.export_name, version, split_vits_ref=args.split_vits_ref)

    # soundfile.write("out.wav", a, vits.hps.data.sampling_rate)
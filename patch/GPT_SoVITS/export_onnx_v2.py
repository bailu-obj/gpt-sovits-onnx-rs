import sys
sys.path.append('./')
import torch
import torchaudio
from torch import nn
from text import cleaned_text_to_sequence
import soundfile
import os
import json
from module.models_onnx import SynthesizerTrn, symbols_v1, symbols_v2
from module.attentions_onnx import MultiHeadAttention
import argparse
from torch import Tensor
import torch.nn.functional as F

from torchaudio.compliance import kaldi as Kaldi
from process_ckpt import get_sovits_version_from_path_fast

V2PRO_SET = {"v2Pro", "v2ProPlus"}

# PyTorch 2.6+ defaults to dynamo/torch.export ONNX; GPT-SoVITS needs legacy export.
_LEGACY_ONNX_KWARGS = {"dynamo": False}


def export_vits_graph(model, inputs, path):
    import onnx

    names = ["text_seq", "pred_semantic", "ref_audio"]
    if len(inputs) == 4:
        names.append("sv_emb")
    torch.onnx.export(
        model, inputs, path, input_names=names, output_names=["audio"],
        dynamic_axes={
            "text_seq": {1: "text_length"},
            "pred_semantic": {2: "pred_length"},
            "ref_audio": {1: "audio_length"},
        },
        opset_version=20, verbose=False, **_LEGACY_ONNX_KWARGS,
    )
    graph = onnx.load(path)
    graph.metadata_props.add(key="vits.unpadded", value="1")
    onnx.save(graph, path)


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
    from sv import SV
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
        global sample
        from AR.models.t2s_model_onnx import sample
        from AR.models.t2s_lightning_module_onnx import Text2SemanticLightningModule
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
        bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1)
        bert = bert.unsqueeze(0)
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

    def export_shared_step(self, path):
        from AR.models.t2s_model_onnx import T2SSharedCacheDecoder

        layers = self.t2s_model.num_layers
        attention = self.t2s_model.h.layers[0].self_attn
        shape = (1, attention.num_heads, 8, attention.head_dim)
        keys = [torch.zeros(shape) for _ in range(layers)]
        values = [torch.zeros(shape) for _ in range(layers)]
        inputs = [f"{kind}_cache_{i}" for kind in ("ik", "iv") for i in range(layers)]
        outputs = [f"{kind}_cache_{i}" for kind in ("k", "v") for i in range(layers)]
        torch.onnx.export(
            T2SSharedCacheDecoder(self.stage_decoder).eval(),
            (torch.zeros((1, 1), dtype=torch.long), keys, values,
             torch.tensor([3]), torch.tensor([5], dtype=torch.int32)),
            path,
            input_names=["iy", *inputs, "position", "past_length"],
            output_names=["logits", *outputs],
            dynamic_axes={name: {2: "cache_capacity"} for name in (*inputs, *outputs)},
            opset_version=20, custom_opsets={"com.microsoft": 1}, **_LEGACY_ONNX_KWARGS,
        )

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name, shared_kv=False):
        torch.onnx.export(
            self.onnx_encoder,
            (ssl_content),
            f"onnx/{project_name}/t2s_encoder.onnx",
            input_names=["ssl_content"],
            output_names=["prompts"],
            dynamic_axes={
                "ssl_content": {2: "ssl_length"},
            },
            opset_version=20,
            **_LEGACY_ONNX_KWARGS,
        )
        
        prompts = self.onnx_encoder(ssl_content)

        bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1)
        bert = bert.unsqueeze(0)
        x = torch.cat([ref_seq, text_seq], 1)
        y_len = prompts.shape[1]
        num_layers = self.t2s_model.num_layers

        # Export first stage decoder
        torch.onnx.export(
            self.first_stage_decoder,
            (x, prompts, bert),
            f"onnx/{project_name}/t2s_fs_decoder.onnx",
            input_names=["x", "prompts", "bert"],
            output_names=["logits"] + [f"k_cache_{i}" for i in range(num_layers)] + 
                         [f"v_cache_{i}" for i in range(num_layers)],
            dynamic_axes={
                "x": {1: "x_length"},
                "prompts": {1: "prompts_length"},
                "bert": {2: "bert_length"},
            },
            verbose=False,
            opset_version=20,
            **_LEGACY_ONNX_KWARGS,
        )
        if shared_kv:
            self.export_shared_step(f"onnx/{project_name}/t2s_s_decoder.onnx")
            return
        # The step tracer only needs the cache layout and one valid token.
        cache_shape = (x.shape[0], x.shape[1] + y_len, self.t2s_model.model_dim)
        k_cache = [bert.new_zeros(cache_shape) for _ in range(num_layers)]
        v_cache = [bert.new_zeros(cache_shape) for _ in range(num_layers)]
        y = torch.cat([prompts, prompts.new_zeros((prompts.shape[0], 1))], dim=1)
        idx = 0
        # Export stage decoder
        torch.onnx.export(
            self.stage_decoder,
            (y, k_cache, v_cache, y_len, idx),
            f"onnx/{project_name}/t2s_s_decoder.onnx",
            input_names=["iy"] + [f"ik_cache_{i}" for i in range(num_layers)] + 
                        [f"iv_cache_{i}" for i in range(num_layers)] + ["y_len", "idx"],
            output_names=["logits"] + [f"k_cache_{i}" for i in range(num_layers)] + 
                         [f"v_cache_{i}" for i in range(num_layers)],
            dynamic_axes={
                "iy": {1: "iy_length"},
                **{f"ik_cache_{i}": {1: "kv_length"} for i in range(num_layers)},
                **{f"iv_cache_{i}": {1: "kv_length"} for i in range(num_layers)},
            },
            verbose=False,
            opset_version=20,
            **_LEGACY_ONNX_KWARGS,
        )
        

class VitsModel(nn.Module):
    def __init__(self, vits_path, version="v2", *, pack_upsamples=True):
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
        if pack_upsamples:
            self.vq_model.dec.pack_upsamples_for_export()
        for layer in self.vq_model.enc_p.modules():
            if isinstance(layer, MultiHeadAttention):
                layer.pack_projections_for_export()
        
    def forward(self, text_seq, pred_semantic, ref_audio, sv_emb=None):
        refer = spectrogram_torch(
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False
        )
        return self.vq_model(pred_semantic, text_seq, refer, sv_emb=sv_emb)[0]

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

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content, project_name, shared_kv=False):
        self.t2s.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name, shared_kv)
        # Use the same representative semantic length as the VITS-only exporter.
        pred_semantic = text_seq.new_zeros((1, 1, 160))
        if is_v2pro(self.version):
            dummy_audio_16k = torchaudio.functional.resample(ref_audio, self.vits.hps.data.sampling_rate, 16000).float()
            audio_feature = Kaldi.fbank(dummy_audio_16k, num_mel_bins=80, sample_frequency=16000, dither=0)
            print("Exporting SV model...")
            print(audio_feature.shape)
            sv_emb = ref_audio.new_zeros((1, self.vits.vq_model.sv_emb.in_features))
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
            inputs = (text_seq, pred_semantic, ref_audio, sv_emb)
        else:
            inputs = (text_seq, pred_semantic, ref_audio)
        export_vits_graph(self.vits, inputs, f"onnx/{project_name}/vits.onnx")

class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        from feature_extractor import cnhubert
        cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.ssl = cnhubert.get_model().model

    def forward(self, ref_audio_16k):
        return self.ssl(ref_audio_16k)["last_hidden_state"].transpose(1, 2)
    


class ExportERes2NetV2(nn.Module): # SV model
    def __init__(self, sv_cn_model: "SV"):
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




class MyBertModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        # The voice checkpoint consumes the state two layers before the end.
        self.bert.encoder.layer = self.bert.encoder.layer[:-2]

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return output.last_hidden_state[0, 1:-1]


def export_bert(project_name):
    from transformers import AutoModel, AutoTokenizer
    bert_path = os.environ.get(
        "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    )
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    inputs = tokenizer("问木兰在想什么，问木兰在惦记什么。", return_tensors="pt")
    bert_model = AutoModel.from_pretrained(bert_path, add_pooling_layer=False)
    my_bert_model = MyBertModel(bert_model).eval()

    torch.onnx.export(
        my_bert_model,
        (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
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


def export(vits_path, gpt_path, project_name, vits_model="v2", shared_kv=False, *, pack_upsamples=True):
    vits = VitsModel(vits_path, version=vits_model, pack_upsamples=pack_upsamples)
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
    gpt_sovits.export(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content, project_name, shared_kv)

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



def export_vits_only(vits_path, project_name, version, *, pack_upsamples=True):
    """Refresh the vocoder from its checkpoint without touching T2S/text models."""
    model = VitsModel(vits_path, version=version, pack_upsamples=pack_upsamples)
    model.eval()
    text = torch.zeros((1, 24), dtype=torch.long)
    semantic = torch.zeros((1, 1, 160), dtype=torch.long)
    reference = torch.zeros((1, model.hps.data.sampling_rate * 5))
    inputs = (text, semantic, reference)
    if is_v2pro(version):
        inputs += (torch.zeros((1, model.vq_model.sv_emb.in_features)),)
    os.makedirs(f"onnx/{project_name}", exist_ok=True)
    export_vits_graph(model, inputs, f"onnx/{project_name}/vits.onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--export_name", type=str, required=True, help="Output directory name; ONNX filenames have no prefix")
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
    parser.add_argument("--gpt-checkpoint", help="Override model_path/gpt.ckpt")
    parser.add_argument("--sovits-checkpoint", help="Override model_path/sovits.pth")
    parser.add_argument("--vits-only", action="store_true", help="Refresh only VITS from checkpoint")
    parser.add_argument("--shared-kv", action="store_true", help="Export the T2S step decoder with reusable CPU KV buffers")
    parser.add_argument("--vits-conv-transpose", action="store_true", help="Keep transposed convolutions in the VITS upsampler")
    args = parser.parse_args()
    if args.vits_only and args.shared_kv:
        parser.error("--shared-kv requires a full export")

    try:
        os.mkdir("onnx")
    except:
        pass
    gpt_path = args.gpt_checkpoint or os.path.join(args.model_path, "gpt.ckpt")
    vits_path = args.sovits_checkpoint or os.path.join(args.model_path, "sovits.pth")
    version = resolve_version(vits_path, args.version, args.auto_version)
    with torch.no_grad():
        if args.vits_only:
            export_vits_only(vits_path, args.export_name, version, pack_upsamples=not args.vits_conv_transpose)
        else:
            export(vits_path, gpt_path, args.export_name, version, args.shared_kv,
                   pack_upsamples=not args.vits_conv_transpose)

    # soundfile.write("out.wav", a, vits.hps.data.sampling_rate)

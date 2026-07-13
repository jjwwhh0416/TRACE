from argparse import Namespace
from pdb import set_trace
import torch
from torch import nn

from src.common import TASKS
from src.data.base import TimeseriesOutputs
from src.utils.masking import Masking
from src.utils.tools import NamespaceWithDefaults, MultiHeadWrapper

from src.models.layers.embed import TimeEmbedding
from src.models.layers.revin import RevIN
from src.models.layers.prediction_head import ClassificationHead, ForecastingHead, ReconstructionHead, EmbeddingHead, RetrievalAugmentedHead, GlobalReconstructionHead, GlobalClassificationHead
from src.models.layers.get_encoder import get_transformer_backbone

class TS_Encoder(nn.Module):
    def __init__(self, configs: Namespace | dict, **kwargs: dict):
        super().__init__()
        configs = self._update_inputs(configs, **kwargs)
        self.configs = configs
        #encoder type
        self.encoder_type = configs.getattr("encoder_type", "patchTST")

        #piplines
        self.chronos_1_pipline = None
        self.chronos_2_pipline = None

        self.task_name = configs.task_name
        self.n_channels = configs.n_channels  # number of channels
        self.output_attention = configs.output_attention
        
        ## Patching parameters
        self.seq_len_channel = configs.seq_len_channel  # length of per channel time-series
        self.patch_len = configs.patch_len  # length of each patch 
        self.patch_stride_len = configs.patch_stride_len  # stride length of each patch
        self.num_patches = (max(self.seq_len_channel, self.patch_len) - self.patch_len) // self.patch_stride_len + 1
        # self.total_len = self.seq_len_channel * self.n_channels + self.n_channels + 1
        
        self.channel_special_tokens = (configs.model_name == "TraceEncoder" and self.encoder_type == "patchTST")
        self.dec_shape = "BTD" if (configs.model_name == "TraceEncoder" and self.encoder_type == "patchTST") else "else"
        # Normalization, patching and embedding
        self.normalizer = RevIN(
            num_features=1, affine=configs.getattr("revin_affine", False)
        )
        self.patch_embedding = TimeEmbedding(
            d_model=configs.d_model,
            num_channels=configs.n_channels,
            patch_len=configs.patch_len,
            stride=configs.patch_stride_len,
            dropout=configs.getattr("dropout", 0.1),
            pos_embed_type=configs.getattr("pos_embed_type", "rel_pos"),
            value_embedding_bias=configs.getattr("value_embedding_bias", False),
            orth_gain=configs.getattr("orth_gain", 1.41),
            channel_special_tokens=self.channel_special_tokens
        )
        self.mask_generator = Masking(mask_ratio=configs.getattr("mask_ratio", 0.0), 
                                      patch_len=configs.patch_len, 
                                      stride=configs.patch_stride_len)

        # Transformer backbone
        self.d_model = configs.d_model
        self.encoder = get_transformer_backbone(configs)  

        if self.encoder_type in ["TimesNet", "TS2Vec", "TimeMixer", "TOTEM"]:
            
            if self.encoder_type == "TimesNet":
                from src.models.TimesNet import Model as TimesNetModel
                configs.task_name = "imputation"
                configs.seq_len = configs.seq_len_channel
                configs.pred_len = configs.getattr("forecast_horizon", 0)
                configs.enc_in = configs.n_channels
                configs.c_out = configs.n_channels
                configs.d_ff = configs.getattr("d_ff", configs.d_model * 4)
                configs.top_k = configs.getattr("top_k", 5)
                configs.num_kernels = configs.getattr("num_kernels", 6)
                
                configs.label_len = 0 
                configs.moving_avg = 25        
                configs.factor = 1            
                configs.dropout = configs.getattr("dropout", 0.1)
                configs.embed = 'timeF'        
                configs.freq = 'h'             
                configs.output_attention = False 
                
                self.local_encoder = TimesNetModel(configs)
                
            elif self.encoder_type == "TS2Vec":
                from src.models.ts2vec import TS2Vec
                input_dims = configs.n_channels
                output_dims = configs.d_model
                hidden_dims = configs.getattr("d_ff", 64)
                depth = configs.getattr("e_layers", 10)
                
                self.local_encoder = TS2Vec(
                    input_dims=input_dims, 
                    output_dims=output_dims, 
                    hidden_dims=hidden_dims, 
                    depth=depth
                ) 
                
            elif self.encoder_type == "TimeMixer":
                from src.models.TimeMixer import Model as TimeMixerModel
                
                configs.task_name = "imputation"
                configs.seq_len = configs.seq_len_channel
                configs.pred_len = configs.getattr("forecast_horizon", 0)
                configs.label_len = 0
                configs.enc_in = configs.n_channels
                configs.dec_in = configs.n_channels
                configs.c_out = configs.n_channels
                configs.d_model = configs.getattr("d_model", 384)
                configs.d_ff = configs.getattr("d_ff", configs.d_model * 4)
                configs.e_layers = configs.getattr("e_layers", 3)
                configs.d_layers = configs.getattr("d_layers", 1)
                configs.n_heads = configs.getattr("n_heads", 8)
                
                configs.down_sampling_layers = configs.getattr("down_sampling_layers", 3)
                configs.down_sampling_window = configs.getattr("down_sampling_window", 2)
                configs.down_sampling_method = configs.getattr("down_sampling_method", "avg")
                configs.channel_independence = configs.getattr("channel_independence", 1)
                configs.use_future_temporal_feature = configs.getattr("use_future_temporal_feature", 0)
                
                configs.decomp_method = configs.getattr("decomp_method", "moving_avg")
                configs.moving_avg = configs.getattr("moving_avg", 25)
                configs.use_norm = configs.getattr("use_norm", 1)
                
                configs.drop_path = configs.getattr("drop_path", 0.1)
                configs.dropout = configs.getattr("dropout", 0.1)
                configs.factor = configs.getattr("factor", 1)
                configs.embed = configs.getattr("embed", 'timeF')
                configs.freq = configs.getattr("freq", 'h')
                configs.activation = configs.getattr("activation", "gelu")
                configs.output_attention = configs.getattr("output_attention", False)
                
                self.local_encoder = TimeMixerModel(configs)
                
            elif self.encoder_type == "TOTEM":
                from pypots.imputation import TOTEM
                
                # 16의 배수(192)로 강제 보정
                self.totem_seq_len = ((configs.seq_len_channel + 15) // 16) * 16
                
                self.local_encoder = TOTEM(
                    n_steps=self.totem_seq_len,  # 186 대신 보정된 192를 전달
                    n_features=configs.n_channels,
                    d_block_hidden=configs.getattr("d_model", 384),          
                    n_residual_layers=configs.getattr("e_layers", 3),        
                    d_residual_hidden=configs.getattr("d_ff", configs.d_model * 4), 
                    d_embedding=configs.getattr("d_model", 384),             
                    n_embeddings=512,                                        
                    epochs=1 
                )

        # Prediction Head
        self.head = self._get_head(self.task_name)
        self.embedding_head = EmbeddingHead(self.n_channels)
        
        
    def set_retriever(self, device):
        from src.models.trace_retriever import RetrievalAugmentedWrapper
        self.retriever = RetrievalAugmentedWrapper(device)
        for param in self.retriever.parameters():
            param.requires_grad = False
        self.top_k = self.configs.top_k

    def _update_inputs(
        self, configs: Namespace | dict, **kwargs
    ) -> NamespaceWithDefaults:
        if isinstance(configs, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**configs, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(configs)


    def _get_head(self, task_name: str) -> nn.Module:
        if hasattr(self.configs, "data_name") and self.configs.data_name in ["health", "env", "energy"]:
            return MultiHeadWrapper({
                "reconstruct_head": ReconstructionHead(
                    self.configs.n_channels,
                    self.configs.d_model,
                    self.configs.patch_len,
                    self.configs.getattr("dropout", 0.1),
                    self.configs.getattr("orth_gain", 1.41),
                ),
                "forecasting_head": ForecastingHead(
                    self.configs.n_channels,
                    self.configs.d_model,
                    self.num_patches,
                    self.configs.forecast_horizon,
                    self.configs.getattr("head_dropout", 0.1),
                )
            })
        else:
            if task_name == TASKS.PRETRAINING:
                if self.encoder_type in ["TimesNet", "TS2Vec", "TimeMixer", "TOTEM"]:
                    return MultiHeadWrapper({
                        "reconstruct_head": GlobalReconstructionHead(
                            self.configs.n_channels,
                            self.configs.d_model,
                            self.configs.patch_len,
                            self.configs.getattr("dropout", 0.1)
                        ),
                        "classification_head": GlobalClassificationHead(
                            self.configs.d_model,
                            self.configs.num_class,
                            self.configs.getattr("dropout", 0.1)
                        )
                    })
                else :
                    return MultiHeadWrapper({
                        "reconstruct_head": ReconstructionHead(
                            self.configs.n_channels,
                            self.configs.d_model,
                            self.configs.patch_len,
                            self.configs.getattr("dropout", 0.1),
                            self.configs.getattr("orth_gain", 1.41),
                        ),
                        "classification_head": ClassificationHead(
                            self.configs.n_channels,
                            self.configs.d_model,
                            self.configs.num_class,
                            self.configs.getattr("dropout", 0.1),
                            self.configs.getattr("view", "global"),
                        )
                    })
            elif task_name == TASKS.RECONSTRUCTION:
                return ReconstructionHead(
                    self.configs.d_model,
                    self.configs.patch_len,
                    self.configs.getattr("dropout", 0.1),
                    self.configs.getattr("orth_gain", 1.41),
                )
            elif task_name == TASKS.CLASSIFICATION:
                return ClassificationHead(
                    self.configs.n_channels,
                    self.configs.d_model,
                    self.configs.num_class,
                    self.configs.getattr("dropout", 0.1),
                    self.configs.getattr("view", "global"),
                )
            elif task_name == TASKS.FORECASTING:
                return ForecastingHead(
                    self.configs.n_channels,
                    self.configs.d_model,
                    self.num_patches,
                    self.configs.forecast_horizon,
                    self.configs.getattr("head_dropout", 0.1),
                )
            elif task_name == TASKS.EMBEDDING:
                return EmbeddingHead(
                    self.configs.n_channels
                )
            elif task_name == TASKS.RAG:
                return RetrievalAugmentedHead(
                    self.configs.n_channels,
                    self.configs.d_model,
                    self.num_patches,
                    self.configs.forecast_horizon,
                    self.configs.getattr("head_dropout", 0.1),
                    self.configs.top_k,
                    self.configs.ts_only
                )
            else:
                raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_encoding_out(self,
        x_enc: torch.Tensor,
        pretrain_mask: torch.Tensor,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [B, C, L] Time-series data
        pretrain_mask  : [B, C, L] Data that is masked but still attended to via mask-tokens
        input_mask : [B, C, L]
        """
        B, C, L = x_enc.shape

        #Normalization
        x_enc = self.normalizer(x=x_enc, mask=pretrain_mask * input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        if (self.encoder_type == "patchTST"):
            # Patching and embedding
            enc_in = self.patch_embedding(x_enc, mask=pretrain_mask)
            # Encoder
            attention_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)  #[B, C, N]
            enc_out, attns = self.encoder(
                x=enc_in,
                attn_mask=attention_mask,
                **{
                    "n_vars": self.n_channels,
                    "n_tokens": self.num_patches,
                }
            )

        elif (self.encoder_type == "Chronos1"):
            current_device = x_enc.device
            print("chronos activated")
            if (self.chronos_1_pipline == None):
                from chronos import ChronosPipeline
                self.chronos_1_pipline = ChronosPipeline.from_pretrained(
                    "amazon/chronos-t5-base",
                    device_map = current_device,
                    torch_dtype = torch.bfloat16
                )
            
            x_reshaped = x_enc.view(B * C, L)
            enc_out_reshaped, _ = self.chronos_1_pipline.embed(x_reshaped.cpu())

            enc_out = enc_out_reshaped.view(B, C, -1, enc_out_reshaped.size(-1))
            enc_out = enc_out.to(device=current_device, dtype=torch.float32)

            import torch.nn.functional as F
            enc_out = F.adaptive_avg_pool2d(enc_out, (self.num_patches, self.d_model))
            attns = None
            print("chronos done")

        elif (self.encoder_type == "Chronos2"):
            print("Chronos2 activated")
            current_device = x_enc.device
            
            if (self.chronos_2_pipline == None):
                from chronos import BaseChronosPipeline
                self.chronos_2_pipline = BaseChronosPipeline.from_pretrained(
                    "amazon/chronos-2", 
                    device_map = current_device,
                    torch_dtype = torch.bfloat16
                )
            
            x_reshaped = x_enc.view(B * C, 1, L)
            
            enc_out_raw = self.chronos_2_pipline.embed(x_reshaped.cpu())
            
            def to_tensor_filtered(item):
                if isinstance(item, (list, tuple)):
                    tensors = [to_tensor_filtered(x) for x in item]
                    valid_tensors = [t for t in tensors if t is not None and t.shape[-1] >= 768]
                    if not valid_tensors: return None
                    return torch.cat(valid_tensors, dim=1)
                
                t = item.clone().detach()
                if t.shape[-1] == 768: 
                    return t.unsqueeze(0) if t.dim() == 2 else t
                return None 

            enc_out_reshaped = to_tensor_filtered(enc_out_raw)
            
            enc_out = enc_out_reshaped.view(B, C, -1, enc_out_reshaped.size(-1))
            enc_out = enc_out.to(device=current_device, dtype=torch.float32)
            
            import torch.nn.functional as F
            enc_out = F.adaptive_avg_pool2d(enc_out, (self.num_patches, self.d_model))
            attns = None

        elif (self.encoder_type == "MOMENT"):
            print("MOMENT activated")
            
            # 💡 [핵심 방어막] 차원 충돌 원천 봉쇄: MOMENT 사용 시 RevIN 바이패스
            if hasattr(self, "normalizer") and not hasattr(self, "_patched_revin"):
                if hasattr(self.normalizer, "_denormalize"):
                    # 기존 함수를 보존하되, MOMENT일 경우 통과시키도록 재정의
                    def safe_denorm_bypass(x_in):
                        # RevIN의 상태(저장된 통계값)를 건드리지 않고,
                        # MOMENT의 출력을 그대로 반환하여 차원 불일치 에러 방지
                        return x_in 
                    self.normalizer._denormalize = safe_denorm_bypass
                self._patched_revin = True

            if not hasattr(self, "moment_model"):
                from momentfm import MOMENTPipeline
                current_device = x_enc.device
                self.moment_model = MOMENTPipeline.from_pretrained(
                    "AutonLab/MOMENT-1-base", 
                    model_kwargs={"task_name": "embedding"}
                )
                self.moment_model.to(current_device)
                self.moment_model.train() 
                self.moment_model.task_name = "embedding"

            # 독립 채널 입력 구성
            B, L, C = x_enc.shape
            x_reshaped = x_enc.transpose(1, 2).reshape(B * C, 1, L)
            
            # 8의 배수로 패딩
            pad_len = 8 - (L % 8) if (L % 8) != 0 else 0
            if pad_len > 0:
                import torch.nn.functional as F
                x_input_padded = F.pad(x_reshaped, (0, pad_len))
            else:
                x_input_padded = x_reshaped

            # 모델 통과
            moment_outputs = self.moment_model(x_enc=x_input_padded)
            
            # 임베딩 추출 안전장치
            enc_out = getattr(moment_outputs, "embeddings", None)
            if enc_out is None:
                enc_out = getattr(moment_outputs, "features", None) or getattr(moment_outputs, "reconstruction", None)
                if enc_out is None:
                    for v in moment_outputs.__dict__.values():
                        if isinstance(v, torch.Tensor):
                            enc_out = v
                            break

            enc_out = enc_out.to(dtype=torch.float32)

            if enc_out.dim() == 2:
                enc_out = enc_out.unsqueeze(1)
            elif enc_out.dim() == 3 and enc_out.shape[1] > enc_out.shape[2]: 
                enc_out = enc_out.transpose(1, 2)
            
            import torch.nn.functional as F
            if enc_out.shape[-1] != self.d_model:
                pad_size = self.d_model - enc_out.shape[-1]
                if pad_size > 0:
                    enc_out = F.pad(enc_out, (0, pad_size))
                else:
                    enc_out = enc_out[..., :self.d_model]

            enc_out = enc_out.view(B, C, -1, self.d_model)
            enc_out = F.adaptive_avg_pool2d(enc_out, (self.num_patches, self.d_model))
            attns = None

        elif (self.encoder_type == "MOIRAI"):
            print("MOIRAI activated")
            
            # RevIN 우회
            if hasattr(self, "normalizer") and not hasattr(self, "_patched_revin"):
                if hasattr(self.normalizer, "_denormalize"):
                    def safe_denorm_bypass(x_in): return x_in 
                    self.normalizer._denormalize = safe_denorm_bypass
                self._patched_revin = True

            if not hasattr(self, "moirai_model"):
                from uni2ts.model.moirai import MoiraiModule
                current_device = x_enc.device
                self.moirai_model = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-base")
                self.moirai_model.to(current_device)
                self.moirai_model.train()

            B, C, L = x_enc.shape
            B_flat = B * C
            P_moirai = 128  
            
            pad_len = (P_moirai - (L % P_moirai)) % P_moirai
            import torch.nn.functional as F
            if pad_len > 0:
                x_enc_padded = F.pad(x_enc, (0, pad_len))
            else:
                x_enc_padded = x_enc
                
            L_padded = x_enc_padded.shape[2]
            target = x_enc_padded.view(B_flat, L_padded, 1)
            
            observed_mask = torch.ones((B_flat, L_padded, 1), device=x_enc.device, dtype=torch.bool)
            if pad_len > 0:
                observed_mask[:, -pad_len:, :] = False 
                
            prediction_mask = torch.zeros((B_flat, L_padded), device=x_enc.device, dtype=torch.bool)
            
            sample_id = torch.arange(B, device=x_enc.device).unsqueeze(1).repeat(1, C).view(B_flat, 1).repeat(1, L_padded)
            time_id = torch.arange(L_padded, device=x_enc.device).unsqueeze(0).repeat(B_flat, 1)
            variate_id = torch.arange(C, device=x_enc.device).unsqueeze(0).repeat(B, 1).view(B_flat, 1).repeat(1, L_padded)
            
            patch_size = torch.tensor([P_moirai], device=x_enc.device, dtype=torch.long)

            captured_embeds = []
            def capture_hook(module, inputs, output):
                if isinstance(output, tuple):
                    captured_embeds.append(output[0])
                else:
                    captured_embeds.append(output)
            
            hook_handle = self.moirai_model.encoder.register_forward_hook(capture_hook)

            _ = self.moirai_model(
                target=target,
                observed_mask=observed_mask,
                prediction_mask=prediction_mask,
                sample_id=sample_id,
                time_id=time_id,
                variate_id=variate_id,
                patch_size=patch_size
            )
            
            hook_handle.remove()
            enc_out = captured_embeds[0].to(dtype=torch.float32)

            if enc_out.dim() == 2: 
                enc_out = enc_out.unsqueeze(1)
                
            enc_out = enc_out.view(B, C, -1, enc_out.shape[-1])
            
            if enc_out.shape[-1] != self.d_model:
                pad_size_dim = self.d_model - enc_out.shape[-1]
                if pad_size_dim > 0: 
                    enc_out = F.pad(enc_out, (0, pad_size_dim))
                else: 
                    enc_out = enc_out[..., :self.d_model]

            enc_out = F.adaptive_avg_pool2d(enc_out, (self.num_patches, self.d_model))
            attns = None

        elif (self.encoder_type in ["TimesNet", "TS2Vec", "TimeMixer", "TOTEM"]):
            print(f"{self.encoder_type} activated")
            
            x_input = x_enc.transpose(1, 2) 

            if self.encoder_type in ["TimeMixer", "TimesNet"]:
                B_val, L_val, C_val = x_input.shape
                dummy_x_mark_enc = torch.zeros(B_val, L_val, 4, device=x_input.device)
                dummy_x_dec = torch.zeros(B_val, 0, C_val, device=x_input.device)
                dummy_x_mark_dec = torch.zeros(B_val, 0, 4, device=x_input.device)
                dummy_mask = torch.ones_like(x_input)
                
                try:
                    enc_out_raw = self.local_encoder(
                        x_input, 
                        dummy_x_mark_enc, 
                        dummy_x_dec, 
                        dummy_x_mark_dec, 
                        mask=dummy_mask
                    )
                except TypeError:
                    enc_out_raw = self.local_encoder(x_input)
                    
            elif self.encoder_type == "TOTEM":
                pad_len = self.totem_seq_len - L
                if pad_len > 0:
                    import torch.nn.functional as F
                    x_input_padded = F.pad(x_input, (0, 0, 0, pad_len))
                else:
                    x_input_padded = x_input
                    
                dummy_mask = torch.ones_like(x_input_padded)
                pypots_inputs = {"X": x_input_padded, "missing_mask": dummy_mask}
                
                totem_outputs = self.local_encoder.model(pypots_inputs)
            
                if isinstance(totem_outputs, dict):
                    enc_out_raw = totem_outputs.get("imputed_data", totem_outputs.get("reconstruction"))
                else:
                    enc_out_raw = totem_outputs

                if pad_len > 0 and enc_out_raw is not None:
                    enc_out_raw = enc_out_raw[:, :L, :]
                
            elif self.encoder_type == "TS2Vec":
                print("TS2Vec activated")
                
                x_input_np = x_input.detach().cpu().numpy()
                
                if hasattr(self.local_encoder, 'encode'):
                    enc_out_raw_np = self.local_encoder.encode(x_input_np)
                else:
                    enc_out_raw_np = self.local_encoder(x_input_np)
                
                enc_out_raw = torch.from_numpy(enc_out_raw_np).to(x_input.device).float()
            else:
                enc_out_raw = self.local_encoder(x_input) 

            if isinstance(enc_out_raw, tuple):
                enc_out_raw = enc_out_raw[0]

            enc_out = enc_out_raw.to(dtype=torch.float32)
            
            if enc_out.shape[-1] != self.d_model:
                pad_size = self.d_model - enc_out.shape[-1]
                if pad_size > 0:
                    import torch.nn.functional as F
                    enc_out = F.pad(enc_out, (0, pad_size))
                else:
                    enc_out = enc_out[:, :, :self.d_model]

            import torch.nn.functional as F
            enc_out = enc_out.transpose(1, 2) 
            enc_out = F.adaptive_avg_pool1d(enc_out, self.num_patches) 
            enc_out = enc_out.transpose(1, 2) 
            
            attns = None
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")
            
        return enc_out, attns
    
    
    def embed(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        """
        x_enc : [B, C, L] Time-series data
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        """
        pretrain_mask = torch.ones_like(input_mask)
        enc_out, attns = self._get_encoding_out(x_enc, pretrain_mask, input_mask)
        
        # Decoder
        input_mask_patch_view = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        emb_dict= self.head(enc_out, input_mask_patch_view, shape=self.dec_shape)
        

        return TimeseriesOutputs(
            input_mask=input_mask,
            embeddings=emb_dict["global"], # [B, d_model]
            channel_embeddings=emb_dict["channels"], # [B, C, d_model]
            cls_embedding=emb_dict["cls"], # [B, d_model]
        )

    def pretraining(
        self,
        x_enc: torch.Tensor,
        pretrain_mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [B, C, L] Time-series data
        pretrain_mask  : [B, C, L] Data that is masked but still attended to via mask-tokens
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        """
        if pretrain_mask is None:
            pretrain_mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            pretrain_mask = pretrain_mask.to(x_enc.device)  # mask: [B, C, L]
        
        enc_out, attns = self._get_encoding_out(x_enc, pretrain_mask, input_mask)
        
        # Decoder
        input_mask_patch_view = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        dec_out = self.head["reconstruct_head"](enc_out, shape=self.dec_shape)  # [B, C, L]
        class_out = self.head["classification_head"](enc_out, input_mask_patch_view, shape=self.dec_shape)  # [B, n_classes]
        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")
        illegal_output = (
            self._check_model_weights_for_illegal_values()
            if self.configs.debug
            else None
        )
        if self.output_attention:
            return TimeseriesOutputs(
                input_mask=input_mask,  # [B, C, L]
                reconstruction=dec_out,  # [B, C, L]
                pretrain_mask=pretrain_mask,  # [B, C, L]   
                classification=class_out,  # [B, n_classes]
                illegal_output=illegal_output  # None or True
            ), attns
        else:
            return TimeseriesOutputs(
                input_mask=input_mask,  # [B, C, L]
                reconstruction=dec_out,  # [B, C, L]
                pretrain_mask=pretrain_mask,  # [B, C, L]   
                classification=class_out,  # [B, n_classes]
                illegal_output=illegal_output  # None or True
            )
            
    def timemmd_pretraining(
        self,
        x_enc: torch.Tensor,
        pretrain_mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [B, C, L] Time-series data
        pretrain_mask  : [B, C, L] Data that is masked but still attended to via mask-tokens
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        """
        if pretrain_mask is None:
            pretrain_mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            pretrain_mask = pretrain_mask.to(x_enc.device)  # mask: [B, C, L]
        
        enc_out, attns = self._get_encoding_out(x_enc, pretrain_mask, input_mask)
        
        # Decoder
        reconstruction = self.head["reconstruct_head"](enc_out, shape=self.dec_shape)  # [B, C, L]
        forecasting = self.head["forecasting_head"](enc_out, shape=self.dec_shape)  # z: [B, C, H]

        # De-Normalization
        reconstruction = self.normalizer(x=reconstruction, mode="denorm")  #[B, C, L]
        forecasting = self.normalizer(x=forecasting, mode="denorm")  #[B, C, H]

        return TimeseriesOutputs(
            input_mask=input_mask,  # [B, C, L]
            reconstruction=reconstruction,  # [B, C, L]
            pretrain_mask=pretrain_mask,  # [B, C, L]   
            forecast=forecasting,  # [B, C, H]
        )        
            
    

    def forecast(
        self, x_enc: torch.Tensor, 
        input_mask: torch.Tensor = None, 
        **kwargs
    ):
        """
        x_enc : [B, C, L] Time-series data
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        """
        pretrain_mask = torch.ones_like(input_mask)
        enc_out, attns = self._get_encoding_out(x_enc, pretrain_mask, input_mask)

        # Decoder
        dec_out = self.head(enc_out, shape=self.dec_shape)  # z: [B, C, H]

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")  #[B, C, H]

        return TimeseriesOutputs(
            input_mask=input_mask,
            forecast=dec_out)

    def classification(
        self, x_enc: torch.Tensor, 
        input_mask: torch.Tensor = None, 
        **kwargs
    ):
        """
        x_enc : [B, C, L] Time-series data
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        """
        pretrain_mask = torch.ones_like(input_mask)
        enc_out, attns = self._get_encoding_out(x_enc, pretrain_mask, input_mask)

        # Decoder
        input_mask_patch_view = Masking.convert_seq_to_patch_view(input_mask, self.patch_len) # [B, C, N]
        dec_out = self.head(enc_out, input_mask_patch_view, shape=self.dec_shape) # [B, n_classes]
        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")  #[B,n_classes]

        return TimeseriesOutputs(
            input_mask=input_mask,
            classification=dec_out,
            )

    def rag_forecasting(
        self, x_enc: torch.Tensor, 
        input_mask: torch.Tensor = None, 
        **kwargs
    ):
        """
        x_enc : [B, C, L] Time-series data
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        """
        pretrain_mask = torch.ones_like(input_mask)
        enc_out, attns = self._get_encoding_out(x_enc, pretrain_mask, input_mask)
        soft_prompt = self.retriever(x_enc, input_mask, top_k=self.top_k)
        dec_out = self.head(enc_out,soft_prompt, shape=self.dec_shape)  # z: [B, C, H]
        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")  #[B, C, H]

        return TimeseriesOutputs(
            input_mask=input_mask,
            forecast=dec_out)


    def forward(
        self,
        x_enc: torch.Tensor,
        pretrain_mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        '''
        Input: (L is the length of per-channel time series)
            x_enc: [B, C, L]
            pretrain_mask: [B, C, L]
            input_mask: [B, C, L]
        '''
        if hasattr(self.configs, "data_name") and self.configs.data_name in ["health", "env", "energy"]:
            return self.timemmd_pretraining(x_enc=x_enc, pretrain_mask=pretrain_mask, input_mask=input_mask, **kwargs)
        else:
            if self.task_name == TASKS.PRETRAINING:  #[reconstruction + global classification]
                return self.pretraining(x_enc=x_enc, pretrain_mask=pretrain_mask, input_mask=input_mask, **kwargs)
            elif self.task_name == TASKS.FORECASTING:
                return self.forecast(x_enc=x_enc, input_mask=input_mask, **kwargs)
            elif self.task_name == TASKS.CLASSIFICATION:
                return self.classification(x_enc=x_enc, input_mask=input_mask, **kwargs)
            elif self.task_name == TASKS.EMBEDDING:
                return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)
            elif self.task_name == TASKS.RAG:
                return self.rag_forecasting(x_enc=x_enc, input_mask=input_mask, **kwargs)
            else:
                raise NotImplementedError(f"Task {self.task_name} not implemented.")

    def _check_model_weights_for_illegal_values(self):
        illegal_encoder_weights = (
            torch.stack([torch.isnan(p).any() for p in self.encoder.parameters()])
            .any()
            .item()
        )
        illegal_head_weights = (
            torch.stack([torch.isnan(p).any() for p in self.head.parameters()])
            .any()
            .item()
        )
        illegal_patch_embedding_weights = (
            torch.stack(
                [torch.isnan(p).any() for p in self.patch_embedding.parameters()]
            )
            .any()
            .item()
        )

        return (
            illegal_encoder_weights
            or illegal_head_weights
            or illegal_patch_embedding_weights
        )
        
        
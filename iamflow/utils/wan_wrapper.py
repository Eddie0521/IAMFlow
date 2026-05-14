# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import os
import types
from typing import List, Optional
import torch
from torch import nn

from iamflow.utils.scheduler import SchedulerInterface, FlowMatchScheduler
from iamflow.vendor.wan.modules.tokenizers import HuggingfaceTokenizer
from iamflow.vendor.wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from iamflow.vendor.wan.modules.vae import _video_vae
from iamflow.utils.tinyvae import TAEHV
from iamflow.vendor.wan.modules.t5 import umt5_xxl
from iamflow.vendor.wan.modules.causal_model import CausalWanModel

# Global variable for Wan model path (can be overridden)
WAN_MODEL_PATH = os.environ.get("WAN_MODEL_PATH", "wan_models/Wan2.1-T2V-1.3B")


class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(f"{WAN_MODEL_PATH}/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        # Move text encoder to GPU if available
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.cuda()

        self.tokenizer = HuggingfaceTokenizer(
            name=f"{WAN_MODEL_PATH}/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        token_embedding = getattr(self.text_encoder, "token_embedding", None)
        if token_embedding is not None and hasattr(token_embedding, "weight"):
            return token_embedding.weight.device
        param = next(self.text_encoder.parameters(), None)
        if param is not None:
            return param.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        # ids = ids.to(torch.device('cpu'))
        # mask = mask.to(torch.device('cpu'))
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, use_lightvae=False, vae_path=None):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self._use_tinyvae = use_lightvae

        if self._use_tinyvae:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            tinyvae_candidates = [
                os.environ.get("TINYVAE_PATH"),
                os.path.join(repo_root, "pretrained", "iamflow_models", "tinyvae.pth"),
                f"{WAN_MODEL_PATH}/tinyvaew2_1.pth",
                f"{WAN_MODEL_PATH}/taew2_1.pth",
                f"{WAN_MODEL_PATH}/lightvaew2_1.pth",
            ]
            vae_path = vae_path or next(
                (path for path in tinyvae_candidates if path and os.path.exists(path)),
                tinyvae_candidates[1],
            )
            print(f"[WanVAEWrapper] Using TinyVAE checkpoint: {vae_path}")
            self.model = TAEHV(checkpoint_path=vae_path).eval().requires_grad_(False)
        else:
            vae_path = vae_path or f"{WAN_MODEL_PATH}/Wan2.1_VAE.pth"
            self.model = _video_vae(
                pretrained_path=vae_path,
                z_dim=16,
                pruning_rate=0.0,
            ).eval().requires_grad_(False)
        self._decode_stream_active = False

    def _tinyvae_dtype(self, dtype: torch.dtype, device: torch.device) -> torch.dtype:
        if device.type == 'cpu' and dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return dtype

    def _prepare_tinyvae_tensor(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.device]:
        param = next(self.model.parameters())
        target_device = param.device
        if target_device.type == 'cpu' and tensor.device.type != 'cpu':
            target_device = tensor.device
        target_dtype = self._tinyvae_dtype(tensor.dtype, target_device)
        if param.device != target_device or param.dtype != target_dtype:
            self.model = self.model.to(device=target_device, dtype=target_dtype)
        return tensor.to(device=target_device, dtype=target_dtype), target_device

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        if self._use_tinyvae:
            source_device = pixel.device
            video = pixel.permute(0, 2, 1, 3, 4).contiguous()
            video, _ = self._prepare_tinyvae_tensor(video)
            video = video.add(1.0).mul(0.5).clamp_(0, 1)
            latents = self.model.encode_video(video, parallel=True, show_progress_bar=False).float()
            return latents.to(device=source_device)

        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def _decode_impl(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if self._use_tinyvae:
            source_device = latent.device
            latent_input, _ = self._prepare_tinyvae_tensor(latent.contiguous())
            frames = self.model.decode_video(
                latent_input,
                parallel=True,
                show_progress_bar=False,
            ).float()
            return frames.mul(2.0).sub(1.0).clamp_(-1, 1).to(device=source_device)

        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def reset_decode_stream(self) -> None:
        if self._use_tinyvae:
            self.model.clear_decode_stream()
        else:
            self.model.clear_cache()
        self._decode_stream_active = False

    def begin_decode_stream(self) -> None:
        self.reset_decode_stream()
        self._decode_stream_active = True

    def end_decode_stream(self) -> None:
        self.reset_decode_stream()

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        return self._decode_impl(latent, use_cache=use_cache)

    def decode_to_pixel_stream(self, latent: torch.Tensor, reset: bool = False) -> torch.Tensor:
        if self._use_tinyvae:
            if reset or not self._decode_stream_active:
                self.begin_decode_stream()
            source_device = latent.device
            latent_input, _ = self._prepare_tinyvae_tensor(latent.contiguous())
            frames = self.model.decode_video_stream(
                latent_input,
                show_progress_bar=False,
            ).float()
            return frames.mul(2.0).sub(1.0).clamp_(-1, 1).to(device=source_device)

        if reset or not self._decode_stream_active:
            self.begin_decode_stream()
        return self._decode_impl(latent, use_cache=True)


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            bank_size=3,
            record_interval=3,
            SMA=False,
            sparse_top_k=6,
    ):
        super().__init__()

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"{WAN_MODEL_PATH}/",
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                bank_size=bank_size,
                record_interval=record_interval,
                SMA=SMA,
                sparse_top_k=sparse_top_k,
            )
        else:
            self.model = WanModel.from_pretrained(f"{WAN_MODEL_PATH}/")
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # self.seq_len = 1560 * local_attn_size if local_attn_size != -1 else 32760 # [1, 21, 16, 60, 104]
        self.seq_len = 1560 * local_attn_size if local_attn_size > 21 else 32760 # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        kv_bank: Optional[List[dict]] = None,
        update_bank: Optional[bool] = False,
        q_bank: Optional[bool] = False,
        update_cache: Optional[bool] = False,
        is_recache: Optional[bool] = False,
        iam_bank_length: Optional[int] = 0,
        # SPT (Soft Prompt Transition) parameters
        prev_crossattn_cache: Optional[List[dict]] = None,
        transition_alpha: Optional[float] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                kv_bank=kv_bank,
                update_bank=update_bank,
                q_bank=q_bank,
                update_cache=update_cache,
                is_recache=is_recache,
                iam_bank_length=iam_bank_length,
                prev_crossattn_cache=prev_crossattn_cache,
                transition_alpha=transition_alpha,
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

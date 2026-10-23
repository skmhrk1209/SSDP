import contextlib
import dataclasses
import enum
from collections.abc import Callable
from pathlib import Path
from typing import override

import jaxtyping as jt
import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from scripts.briarmbg import BriaRMBG
from scripts.image_process import prepare_image
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_cluster import fps
from transformers import BitImageProcessor, Dinov2Model
from triposg.models.autoencoders import TripoSGVAEModel, autoencoder_kl_triposg
from triposg.models.autoencoders.vae import DiagonalGaussianDistribution
from triposg.models.transformers import TripoSGDiTModel
from triposg.pipelines.pipeline_triposg import TripoSGPipeline, retrieve_timesteps

from ssdp.utils.jaxtyping import jaxtyped

from .ldm import LDM, LDMConfig

# NOTE: https://github.com/VAST-AI-Research/TripoSG
# > If you want to use the full VAE module (including the encoder part),
# you need to uncomment the Line-15 in triposg/models/autoencoders/autoencoder_kl_triposg.py
# and install torch-cluster. and run:
autoencoder_kl_triposg.fps = fps

type SDFType = Callable[
    [jt.Float[torch.Tensor, " *B 3 "]],
    jt.Float[torch.Tensor, " *B 1 "],
]


class EncoderOutputMode(enum.StrEnum):
    MEAN = enum.auto()
    SAMPLE = enum.auto()


class TransformerDType(enum.StrEnum):
    FLOAT32 = enum.auto()
    FLOAT16 = enum.auto()
    BFLOAT16 = enum.auto()


@dataclasses.dataclass
class TripoSGPriorConfig(LDMConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: TripoSGPrior,
    )
    repo_id: str = "VAST-AI/TripoSG"
    rmbg_repo_id: str = "briaai/RMBG-1.4"
    # https://github.com/VAST-AI-Research/TripoSG/blob/fc5c409/triposg/models/autoencoders/autoencoder_kl_triposg.py#L440
    num_latent_tokens: int = 2048
    # https://github.com/VAST-AI-Research/TripoSG/blob/fc5c409/scripts/inference_vae.py#L44
    num_surface_points: int = 204800
    # https://github.com/VAST-AI-Research/TripoSG/blob/fc5c409/scripts/inference_triposg.py#L77
    guidance_scale: float = 7.0
    # TODO: desctive the evidence
    distance_per_unit: float = 0.0043
    encoder_output_mode: EncoderOutputMode = EncoderOutputMode.MEAN
    transformer_dtype: TransformerDType = TransformerDType.FLOAT16


class TripoSGPrior(LDM):
    config: TripoSGPriorConfig
    aabb: torch.Tensor

    def __init__(self, config: TripoSGPriorConfig) -> None:
        super().__init__(config)
        self.pipeline: TripoSGPipeline = TripoSGPipeline.from_pretrained(
            snapshot_download(config.repo_id),
        )
        self.rmbg_model: BriaRMBG = BriaRMBG.from_pretrained(
            snapshot_download(self.config.rmbg_repo_id),
        )
        self.auto_encoder: TripoSGVAEModel = self.pipeline.vae
        self.transformer: TripoSGDiTModel = self.pipeline.transformer
        self.image_encoder: Dinov2Model = self.pipeline.image_encoder_dinov2
        self.image_processor: BitImageProcessor = self.pipeline.feature_extractor_dinov2
        self.image_embeddings = None
        self.already_conditioned = False
        self.register_buffer("aabb", torch.tensor([[-1.0] * 3, [1.0] * 3]))

    @override
    @property
    def latent_shape(self) -> tuple[int, ...]:
        return (self.config.num_latent_tokens, self.auto_encoder.latent_channels)

    @torch.no_grad()
    def condition(self, image_file: Path | None = None) -> None:
        if image_file is None:
            image_file = self.config.condition_image_file
        image = prepare_image(
            image_path=image_file,
            bg_color=np.ones(3),
            rmbg_net=self.rmbg_model,
        )
        cond_image_embeddings, uncond_image_embeddings = self.pipeline.encode_image(
            image=image,
            device=self.image_encoder.device,
            num_images_per_prompt=1,
        )
        image_embeddings = torch.cat(
            [uncond_image_embeddings, cond_image_embeddings],
            dim=0,
        )
        self.image_embeddings = image_embeddings
        self.already_conditioned = True

    @torch.no_grad()
    def uncondition(self) -> None:
        image = Image.new("RGB", (224, 224))
        _, image_embeddings = self.pipeline.encode_image(
            image=image,
            device=self.image_encoder.device,
            num_images_per_prompt=1,
        )
        self.image_embeddings = image_embeddings
        self.already_conditioned = False

    @jaxtyped()
    @torch.no_grad()
    def encode(
        self,
        mesh: trimesh.Trimesh,
        seed: int | None = None,
    ) -> jt.Float[torch.Tensor, " L C "]:
        positions, face_indices = trimesh.sample.sample_surface(
            mesh=mesh,
            count=self.config.num_surface_points,
            seed=seed,
        )
        normals = mesh.face_normals[face_indices, ...]
        surface = np.concatenate([positions, normals], axis=-1)
        surface = torch.as_tensor(
            data=surface,
            dtype=self.auto_encoder.dtype,
            device=self.auto_encoder.device,
        )
        latent_distribution: DiagonalGaussianDistribution = self.auto_encoder.encode(
            x=surface.unsqueeze(0),
            num_tokens=self.config.num_latent_tokens,
            seed=seed,
        ).latent_dist
        if self.config.encoder_output_mode is EncoderOutputMode.MEAN:
            latents = latent_distribution.mode()
        else:
            generator = None
            if seed is not None:
                generator = torch.Generator(self.auto_encoder.device)
                generator.manual_seed(seed)
            latents = latent_distribution.sample(generator)
        latents = latents.squeeze(0)
        return latents

    @jaxtyped()
    def decode(
        self,
        latents: jt.Float[torch.Tensor, " L C "],
    ) -> SDFType:
        latents = latents.unsqueeze(0)
        latents = self.auto_encoder.post_quant(latents)
        queries = self.auto_encoder.embedder(latents.new_zeros(1, 1, 3))
        _, kv_cache = self.auto_encoder.decoder(latents, queries, None)

        @jaxtyped()
        def sdf(
            positions: jt.Float[torch.Tensor, " *B 3 "],
            sdf_scale: float = self.config.distance_per_unit,
        ) -> jt.Float[torch.Tensor, " *B 1 "]:
            clamped_positions = torch.clamp(positions, *self.aabb)
            sdf_shifts = torch.linalg.vector_norm(
                x=positions - clamped_positions,
                dim=-1,
                keepdim=True,
            )
            queries = self.auto_encoder.embedder(clamped_positions)
            needs_second_derivative = torch.is_grad_enabled() and positions.requires_grad
            attention_kernel = (
                sdpa_kernel(SDPBackend.MATH)
                if needs_second_derivative
                else contextlib.nullcontext()
            )
            with attention_kernel:
                queries = queries.flatten(0, -2).unsqueeze(0)
                tsdf_values, _ = self.auto_encoder.decoder(None, queries, kv_cache)
                tsdf_values = tsdf_values.squeeze(0).unflatten(0, positions.shape[:-1])
            sdf_values = tsdf_values * sdf_scale + sdf_shifts
            return sdf_values

        return sdf

    @jaxtyped()
    @torch.no_grad()
    def tweedie(
        self,
        noisy_latents: jt.Float[torch.Tensor, " L C "],
        noise_level: jt.Float[torch.Tensor, " "],
    ) -> jt.Float[torch.Tensor, " L C "]:
        noisy_latents = noisy_latents.unsqueeze(0)

        # NOTE: convert the noise level from VE-SDE to Rectified Flow.
        noise_level = noise_level / (1.0 + noise_level)
        noisy_latents = (1.0 - noise_level) * noisy_latents

        [timestep], _ = retrieve_timesteps(
            scheduler=self.pipeline.scheduler,
            device=self.transformer.device,
            sigmas=noise_level.unsqueeze(0).cpu().numpy(),
        )

        transformer_dtype = getattr(torch, self.config.transformer_dtype)
        with torch.autocast(
            device_type=self.transformer.device.type,
            dtype=transformer_dtype,
        ):
            hidden_states = (
                torch.cat([noisy_latents, noisy_latents], dim=0)
                if self.already_conditioned
                else noisy_latents
            )
            (velocities,) = self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=self.image_embeddings,
                timestep=timestep.expand(*hidden_states.shape[:1]),
                return_dict=False,
            )

        velocities = velocities.to(torch.float32)

        if self.already_conditioned:
            uncond_velocities, cond_velocities = torch.chunk(velocities, 2, dim=0)
            velocities = torch.lerp(uncond_velocities, cond_velocities, self.config.guidance_scale)

        denoised_latents = noisy_latents + noise_level * velocities

        denoised_latents = denoised_latents.squeeze(0)

        return denoised_latents

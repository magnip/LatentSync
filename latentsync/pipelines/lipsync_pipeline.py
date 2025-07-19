# Adapted from https://github.com/bytedance/LatentSync/blob/76eddc137b15701e40cbdf69f1006cabcce5ad91/latentsync/pipelines/lipsync_pipeline.py

import inspect
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess
import tempfile
import math
import time

import numpy as np
import torch
import torchvision

from diffusers.utils import is_accelerate_available
from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.image_processor import ImageProcessor
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)

class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")
        for cpu_offloaded_model in [self.unet, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(f"`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.")

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_path, video_skip_frames=1, video_num_workers=4):
        video_frames = read_video(video_path, change_fps=False, use_decord=True)
        faces, debug_faces, original_video_frames, boxes, affine_matrices = [], [], [], [], []

        print(f"Transforming {len(video_frames)} faces...")
        for i, frame in enumerate(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            if face is None:
                print(f"Warning: No face detected in frame {i}, retaining original frame")
                face = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                face = (face * 2 - 1).clamp(-1, 1)
                box = [0, 0, frame.shape[1], frame.shape[0]]
                affine_matrix = torch.eye(3)
            faces.append(face)
            debug_faces.append(face)
            original_video_frames.append(frame)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        if faces:
            faces = torch.stack(faces)
        else:
            print("Error: No valid faces detected in any frame. Returning empty tensors.")
            faces = torch.empty((0,))
        print(f"Processed faces: {len(faces)} frames")
        return faces, debug_faces, original_video_frames, boxes, affine_matrices

    def restore_video(self, faces, video_frames, boxes, affine_matrices):
        video_frames = video_frames[: faces.shape[0]]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(faces):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            if width == video_frames[index].shape[1] and height == video_frames[index].shape[0]:
                out_frame = video_frames[index]
            else:
                face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
                face = rearrange(face, "c h w -> h w c")
                face = (face / 2 + 0.5).clamp(0, 1)
                face = (face * 255).to(torch.uint8).cpu().numpy()
                out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        restored_frames = np.stack(out_frames, axis=0)
        print(f"Restored frames: {len(out_frames)}")
        return restored_frames

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        video_skip_frames: int = 1,
        video_num_workers: int = 4,
        **kwargs,
    ):
        overall_start_time = time.time()
        is_train = self.unet.training
        self.unet.eval()
        check_ffmpeg_installed()

        self.image_processor = ImageProcessor(height, mask=mask, device="cuda")
        print("\n" + "="*80)
        print(f"Starting LatentSync pipeline with {num_inference_steps} inference steps")
        print(f"Video: {video_path}")
        print(f"Audio: {audio_path}")
        print(f"Output: {video_out_path}")
        print("="*80 + "\n")

        print("\n[1/5] Extracting audio features...")
        audio_start = time.time()
        audio_samples = read_audio(audio_path)
        audio_time = time.time() - audio_start
        print(f"Audio feature extraction completed in {audio_time:.2f} seconds")

        print("\n[2/5] Processing video frames...")
        video_start = time.time()
        faces, debug_faces, original_video_frames, boxes, affine_matrices = self.affine_transform_video(
            video_path, video_skip_frames=video_skip_frames, video_num_workers=video_num_workers
        )
        video_time = time.time() - video_start
        print(f"Video processing completed in {video_time:.2f} seconds")

        video_duration = len(original_video_frames) / video_fps
        audio_duration = len(audio_samples) / audio_sample_rate
        print(f"Input video duration: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s")
        print(f"Input video frames: {len(original_video_frames)}, Processed faces: {len(faces)}")

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(height, width, callback_steps)

        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
        timesteps = self.scheduler.timesteps
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        self.video_fps = video_fps

        if self.unet.add_audio_layer:
            whisper_feature = self.audio_encoder.audio2feat(audio_path)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
            total_frames = len(faces)  # Force to match video frames
            if len(whisper_chunks) < total_frames:
                print(f"Warning: Whisper chunks ({len(whisper_chunks)}) less than faces ({total_frames}). Padding audio.")
                whisper_chunks.extend([whisper_chunks[-1]] * (total_frames - len(whisper_chunks)))
        else:
            total_frames = len(faces)

        num_inferences = math.ceil(total_frames / num_frames)
        print(f"Number of faces: {len(faces)}, Whisper chunks: {len(whisper_chunks) if self.unet.add_audio_layer else 'N/A'}, Num inferences: {num_inferences}, Total frames: {total_frames}")

        synced_video_frames = []
        num_channels_latents = self.vae.config.latent_channels

        if num_inferences <= 0:
            print("Error: No inferences to perform. Check face detection or audio features.")
            return

        print("\n[3/5] Encoding video frames to latent space...")
        latent_start = time.time()
        all_latents = self.prepare_latents(
            1,
            total_frames,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            self._execution_device,
            generator,
        )
        all_latents = all_latents * self.scheduler.init_noise_sigma
        latent_time = time.time() - latent_start
        print(f"Latent encoding completed in {latent_time:.2f} seconds")

        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            start_idx = i * num_frames
            end_idx = min(start_idx + num_frames, total_frames)
            chunk_size = end_idx - start_idx

            if chunk_size == 0:
                print(f"Warning: Skipping batch {i} due to zero chunk size")
                continue

            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[start_idx:end_idx])
                if audio_embeds.shape[0] != chunk_size:
                    print(f"Warning: Skipping batch {i} due to insufficient audio frames: {audio_embeds.shape[0]}")
                    continue
                audio_embeds = audio_embeds.to(device=self._execution_device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None

            inference_faces = faces[start_idx:end_idx]
            if inference_faces.shape[0] != chunk_size:
                print(f"Warning: Skipping batch {i} due to insufficient video frames: {inference_faces.shape[0]}")
                continue

            latents = all_latents[:, :, start_idx:end_idx]
            if latents.shape[2] != chunk_size:
                print(f"Warning: Skipping batch {i} due to insufficient latents: {latents.shape[2]}")
                continue

            pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                self._execution_device,
                generator,
                do_classifier_free_guidance,
            )

            image_latents = self.prepare_image_latents(
                pixel_values,
                self._execution_device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            if isinstance(self.scheduler, (DDIMScheduler, PNDMScheduler)) and i == 0:
                warmup_steps = min(num_inference_steps + 5, 35)
                self.scheduler.set_timesteps(warmup_steps, device=self._execution_device)
                timesteps = self.scheduler.timesteps
            else:
                self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
                timesteps = self.scheduler.timesteps

            with self.progress_bar(total=len(timesteps)) as progress_bar:
                for j, t in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat(
                        [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                    )
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, pixel_values, 1 - masks, self._execution_device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)
            print(f"Batch {i}: Processed {chunk_size} frames, Total synced frames: {sum(f.shape[0] for f in synced_video_frames)}")

        if not synced_video_frames:
            print("Error: No video frames generated. Check face detection or audio processing.")
            return

        print("\n[4/5] Restoring video...")
        restore_start = time.time()
        synced_video_frames = self.restore_video(
            torch.cat(synced_video_frames), original_video_frames, boxes, affine_matrices
        )
        restore_time = time.time() - restore_start
        print(f"Video restoration completed in {restore_time:.2f} seconds")

        video_length = synced_video_frames.shape[0] / video_fps
        audio_samples_remain_length = int(video_length * audio_sample_rate)
        if audio_samples.shape[0] < audio_samples_remain_length:
            padding = torch.zeros(audio_samples_remain_length - audio_samples.shape[0], device=audio_samples.device)
            audio_samples = torch.cat([audio_samples, padding]).cpu().numpy()
        else:
            audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        print("\n[5/5] Writing output video...")
        output_start = time.time()
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        temp_video = os.path.join(temp_dir, "video.mp4")
        temp_audio = os.path.join(temp_dir, "audio.wav")
        write_video(temp_video, synced_video_frames, fps=video_fps)
        sf.write(temp_audio, audio_samples, audio_sample_rate)

        command = f"ffmpeg -threads 32 -y -loglevel error -nostdin -i {temp_video} -i {temp_audio} -c:v libx264 -r {video_fps} -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True, check=True)

        output_time = time.time() - output_start
        print(f"Output writing completed in {output_time:.2f} seconds")

        total_time = time.time() - overall_start_time
        print("\n" + "="*80)
        print(f"LatentSync pipeline completed in {total_time:.2f} seconds")
        print(f"Output video frames: {synced_video_frames.shape[0]}, Duration: {video_length:.2f}s")
        print(f"Output audio samples: {audio_samples.shape[0]}, Duration: {audio_samples.shape[0] / audio_sample_rate:.2f}s")
        print("="*80)

        if is_train:
            self.unet.train()

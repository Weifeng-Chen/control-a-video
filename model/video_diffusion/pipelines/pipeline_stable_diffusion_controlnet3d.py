
from .pipeline_st_stable_diffusion import SpatioTemporalStableDiffusionPipeline
from typing import Callable, List, Optional, Union
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from transformers import DPTForDepthEstimation
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
import torch
from einops import rearrange, repeat
import decord
import cv2
import random
import numpy as np
from ..models.unet_3d_condition import UNetPseudo3DConditionModel
from ..models.controlnet3d import ControlNet3DModel


class Controlnet3DStableDiffusionPipeline(SpatioTemporalStableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNetPseudo3DConditionModel,
        controlnet: ControlNet3DModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        annotator_model=None,

    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)

        self.annotator_model = annotator_model
        self.controlnet = controlnet
        self.unet = unet
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )

    @staticmethod
    def get_frames_preprocess(data_path, num_frames=24, sampling_rate=1, begin_indice=0, return_np=False):
        vr = decord.VideoReader(data_path,)
        n_images = len(vr)
        fps_vid = round(vr.get_avg_fps())
        frame_indices = [begin_indice + i*sampling_rate for i in range(num_frames)]   # 随机取n帧
        

        while n_images <= frame_indices[-1]:
            # 超过视频长度，采样率减小直至不超过。
            sampling_rate -= 1
            if sampling_rate == 0:
                # NOTE 边界检查
                return None, None
            frame_indices = [i*sampling_rate for i in range(num_frames)]
        frames = vr.get_batch(frame_indices).asnumpy()

        if return_np:
            return frames, fps_vid

        frames = torch.from_numpy(frames).div(255) * 2 - 1
        frames = rearrange(frames, "f h w c -> c f h w").unsqueeze(0)
        return frames, fps_vid  

    @torch.no_grad()
    def get_canny_edge_map(self, frames, ):
        # (b f) c h w"
        # from tensor to numpy
        inputs = frames.cpu().numpy() 
        inputs = rearrange(inputs, 'f c h w -> f h w c')
        # inputs from [-1, 1] to [0, 255]
        inputs = (inputs + 1) * 127.5
        inputs = inputs.astype(np.uint8)
        lower_threshold = 100
        higher_threshold = 200
        edge_images = np.stack([cv2.Canny(inp, lower_threshold, higher_threshold) for inp in inputs])
        # from numpy to tensors
        edge_images = torch.from_numpy(edge_images).unsqueeze(1)   # f, 1, h, w
        edge_images = edge_images.div(255)*2 - 1 
        # print(torch.max(out_images), torch.min(out_images), out_images.dtype)
        return edge_images.to(dtype= self.controlnet.dtype, device=self.controlnet.device)

    @torch.no_grad()
    def get_depth_map(self, frames, height, width, return_standard_norm=False ):
        """
            frames should be like: (f c h w), you may turn b f c h w -> (b f) c h w first
        """
        h,w = height, width
        inputs = torch.nn.functional.interpolate(
            frames,
            size=(384, 384),
            mode="bicubic",
            antialias=True,
        ) 
        # 转类型和设备
        inputs = inputs.to(dtype= self.annotator_model.dtype, device=self.annotator_model.device)

        outputs = self.annotator_model(inputs)
        predicted_depths = outputs.predicted_depth

        # interpolate to original size
        predictions = torch.nn.functional.interpolate(
            predicted_depths.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
        )

        # normalize output
        if return_standard_norm:
            depth_min = torch.amin(predictions, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(predictions, dim=[1, 2, 3], keepdim=True)
            predictions = 2.0 * (predictions - depth_min) / (depth_max - depth_min) - 1.0
        else:
            predictions -= torch.min(predictions)
            predictions /= torch.max(predictions)

        return predictions  


    @torch.no_grad()
    def get_hed_map(self, frames,):
        if isinstance(frames, torch.Tensor):
            # 输入的就是 b c h w的tensor 范围是-1~1，需要转换为0～1
            frames = (frames + 1) / 2
            #rgb转bgr
            bgr_frames = frames.clone()
            bgr_frames[:, 0, :, :] = frames[:, 2, :, :]
            bgr_frames[:, 2, :, :] = frames[:, 0, :, :]

            edge = self.annotator_model(bgr_frames) # 范围也是0～1
            return edge
        else:
            assert frames.ndim == 3
            frames = frames[:, :, ::-1].copy()
            with torch.no_grad():
                image_hed = torch.from_numpy(frames).to(next(self.annotator_model.parameters()).device, dtype=next(self.annotator_model.parameters()).dtype )
                image_hed = image_hed / 255.0
                image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
                edge = self.annotator_model(image_hed)[0]
                edge = (edge.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                return edge[0]
    
    @torch.no_grad()
    def get_pose_map(self, frames,):
        if isinstance(frames, torch.Tensor):
            # 输入的就是 b c h w的tensor 范围是-1~1，需要转换为0～1
            frames = (frames + 1) / 2
            np_frames = frames.cpu().numpy() * 255
            np_frames = np.array(np_frames, dtype=np.uint8)
            np_frames = rearrange(np_frames, 'f c h w-> f h w c')
            poses = np.stack([self.annotator_model(inp) for inp in np_frames])
        else:
            poses = self.annotator_model(frames)
        return poses
    
    def get_timesteps(self, num_inference_steps, strength,):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        controlnet_hint = None,
        fps_labels = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        clip_length: int = 8, # NOTE clip_length和images的帧数一致。
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs = None,
        video_scale: float = 0.0,
        controlnet_conditioning_scale: float = 1.0,
        fix_first_frame=True,
        first_frame_output = None ,  # 也可以允许挑好图后传入。
        first_frame_output_latent = None,
        first_frame_control_hint = None,    # 维持第一帧
        add_first_frame_by_concat = False,
        controlhint_in_uncond = False,
        init_same_noise_per_frame=False,
        init_noise_by_residual_thres=0.0,
        images=None,
        in_domain=False,    # 是否调用视频模型生成图片
        residual_control_steps=1,
        first_frame_ddim_strength=1.0,
        return_last_latent = False,
    ):
        '''
        add origin video frames to get depth maps
        '''
        
        if fix_first_frame and first_frame_output is None and first_frame_output_latent is None:
            first_frame_output = self.__call__(
                prompt=prompt,
                controlnet_hint=controlnet_hint[:,:,0,:,:] if not in_domain else controlnet_hint[:,:,0:1,:,:],
                # b c f h w
                num_inference_steps=20,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                fix_first_frame=False,
                controlhint_in_uncond=controlhint_in_uncond,
            ).images[0]
        

        if first_frame_output is not None:
            if isinstance(first_frame_output, list):
                first_frame_output = first_frame_output[0] 
            first_frame_output = torch.from_numpy(np.array(first_frame_output)).div(255) * 2 - 1
            first_frame_output = rearrange(first_frame_output, "h w c -> c h w").unsqueeze(0)   # FIXME 目前不允许多个batch 先设置为1
            first_frame_output = first_frame_output.to(dtype= self.vae.dtype, device=self.vae.device)

            first_frame_output_latent = self.vae.encode(first_frame_output).latent_dist.sample()
            first_frame_output_latent = first_frame_output_latent * 0.18215
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 5.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        if controlnet_hint is not None:
            if len(controlnet_hint.shape) == 5:
                clip_length = controlnet_hint.shape[2]
            else:
                clip_length = 0
            
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            clip_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype


        if len(latents.shape) == 5 and init_same_noise_per_frame:
            latents[:,:,1:,:,:] = latents[:,:,0:1,:,:]

        if len(latents.shape) == 5 and init_noise_by_residual_thres > 0.0 and images is not None:
            
            images = images.to(device=device, dtype=latents_dtype)  # b c f h w
            image_residual = torch.abs(images[:,:,1:,:,:] - images[:,:,:-1,:,:])
            images = rearrange(images, "b c f h w -> (b f) c h w")
            
            # norm residual
            image_residual = image_residual / torch.max(image_residual)
            
            image_residual = rearrange(image_residual, "b c f h w -> (b f) c h w")
            image_residual = torch.nn.functional.interpolate(
                        image_residual, 
                        size=(latents.shape[-2], latents.shape[-1]),
                        mode='bilinear')
            image_residual = torch.mean(image_residual, dim=1)

            image_residual_mask = (image_residual > init_noise_by_residual_thres).float()
            image_residual_mask = repeat(image_residual_mask, '(b f) h w -> b f h w', b=batch_size)
            image_residual_mask = repeat(image_residual_mask, 'b f h w -> b c f h w', c=latents.shape[1])

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if fix_first_frame:
                if add_first_frame_by_concat:
                    if len(first_frame_output_latent.shape) == 4:
                        latents = torch.cat([first_frame_output_latent.unsqueeze(2), latents], dim=2)
                    else:
                        latents = torch.cat([first_frame_output_latent, latents], dim=2)
                    if first_frame_control_hint is not None:
                        controlnet_hint = torch.cat([first_frame_control_hint, controlnet_hint], dim=2)
                    else:
                        controlnet_hint = torch.cat([controlnet_hint[:,:,0:1 ,:,:], controlnet_hint], dim=2)
            
            if controlhint_in_uncond:
                controlnet_hint = torch.cat([controlnet_hint] * 2) if do_classifier_free_guidance else controlnet_hint
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                if i<residual_control_steps and  len(latents.shape) == 5 and init_noise_by_residual_thres > 0.0 and images is not None :
                    if first_frame_ddim_strength < 1.0  and i == 0 :
                        # NOTE DDIM to get the first noise
                        first_frame_output_latent_DDIM = first_frame_output_latent.clone()
                        full_noise_timestep, _ = self.get_timesteps(num_inference_steps, strength=first_frame_ddim_strength)
                        latent_timestep = full_noise_timestep[:1].repeat(batch_size * num_images_per_prompt)
                        first_frame_output_latent_DDIM = self.scheduler.add_noise(first_frame_output_latent_DDIM, latents[:,:,0,:,:], latent_timestep)
                        latents[:,:,0,:,:]=first_frame_output_latent_DDIM
                    begin_frame = 1
                    for n_frame in range(begin_frame, latents.shape[2]):
                        latents[:,:, n_frame, :, :] = \
                            (latents[:,:, n_frame, :, :] - latents[:,:, n_frame-1, :, :]) \
                            * image_residual_mask[:,:, n_frame-1, :, :] + \
                            latents[:,:, n_frame-1, :, :]
                if fix_first_frame:
                    latents[:,:,0 ,:,:] = first_frame_output_latent

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if controlnet_hint is not None:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=controlnet_hint,
                        return_dict=False,
                    )
                    down_block_res_samples = [
                    down_block_res_sample * controlnet_conditioning_scale
                    for down_block_res_sample in down_block_res_samples
                    ]
                    mid_block_res_sample *= controlnet_conditioning_scale

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample.to(dtype=latents_dtype)
                else:
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                    ).sample.to(dtype=latents_dtype)

                if video_scale > 0 and controlnet_hint is not None:
                    bsz = latents.shape[0]
                    f = latents.shape[2]
                    # 逐帧预测
                    latent_model_input_single_frame = rearrange(latent_model_input, 'b c f h w -> (b f) c h w')
                    text_embeddings_single_frame = torch.cat([text_embeddings] * f, dim=0)
                    control_maps_single_frame = rearrange(controlnet_hint, 'b c f h w -> (b f) c h w')
                    latent_model_input_single_frame = latent_model_input_single_frame.chunk(2, dim=0)[0]
                    text_embeddings_single_frame = text_embeddings_single_frame.chunk(2, dim=0)[0]
                    if controlhint_in_uncond:
                        control_maps_single_frame = control_maps_single_frame.chunk(2, dim=0)[0]

                    down_block_res_samples_single_frame, mid_block_res_sample_single_frame = self.controlnet(
                                latent_model_input_single_frame,
                                t,
                                encoder_hidden_states=text_embeddings_single_frame,
                                controlnet_cond=control_maps_single_frame,
                                return_dict=False,
                            )
                    down_block_res_samples_single_frame = [
                    down_block_res_sample_single_frame * controlnet_conditioning_scale
                    for down_block_res_sample_single_frame in down_block_res_samples_single_frame
                    ]
                    mid_block_res_sample_single_frame *= controlnet_conditioning_scale

                    noise_pred_single_frame_uncond = self.unet(
                                latent_model_input_single_frame,
                                t,
                                encoder_hidden_states = text_embeddings_single_frame,
                                down_block_additional_residuals=down_block_res_samples_single_frame,
                                mid_block_additional_residual=mid_block_res_sample_single_frame,
                                ).sample
                    noise_pred_single_frame_uncond = rearrange(noise_pred_single_frame_uncond, '(b f) c h w -> b c f h w', f=f)
                # perform guidance
                if do_classifier_free_guidance:
                    if video_scale > 0 and controlnet_hint is not None:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_single_frame_uncond + video_scale * (
                            noise_pred_uncond - noise_pred_single_frame_uncond
                        ) + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        # 8. Post-processing
        image = self.decode_latents(latents)
        if add_first_frame_by_concat:
            image = image[:,1:,:,:,:]
            
        # 9. Run safety checker
        has_nsfw_concept = None
        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        if return_last_latent:
            last_latent = latents[:,:,-1,:,:]
            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), last_latent 
        else:
            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

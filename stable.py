import torch
from typing import Optional, Union, List, Dict, Any, Callable
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipelineOutput,
)
from algorithms.chords import CHORDS


@torch.no_grad()
def forward_chords_worker(
    self,
    mp_queues: Optional[torch.FloatTensor] = None,
    device: Optional[str] = None,
    **kwargs,
):
    """
    Worker side for CHORDS on Stable Diffusion (tiny-stable-diffusion-torch).
    Expects a tuple from master:
      (latents, timestep, prompt_embeds, negative_prompt_embeds, guidance_scale, idx)
    Returns:
      (noise_pred, idx)
    """
    while True:
        ret = mp_queues[0].get()
        if ret is None:
            return

        latents, t, prompt_embeds, negative_prompt_embeds, guidance_scale, idx = ret

        # Ensure tensor/device/dtype
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        device = device or self._execution_device
        latents = latents.to(device)
        t = t.to(latents.dtype).to(device)

        # ===== Standard CFG =====
        do_cfg = (guidance_scale is not None and guidance_scale > 1.0
                  and negative_prompt_embeds is not None)

        if do_cfg:
            x_in = torch.cat([latents, latents], dim=0)
            e_in = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device)
            noise = self.unet(x_in, t, encoder_hidden_states=e_in).sample
            e_uncond, e_text = noise.chunk(2, dim=0)
            noise_pred = e_uncond + guidance_scale * (e_text - e_uncond)
        else:
            noise_pred = self.unet(latents, t, encoder_hidden_states=prompt_embeds.to(device)).sample

        # For cross-process safety, send back CPU tensors
        mp_queues[1].put((noise_pred.to("cpu"), idx))


@torch.no_grad()
def forward_chords(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ("latents",),
    num_cores: int = 1,
    mp_queues: Optional[torch.FloatTensor] = None,
    full_return: bool = False,
    init_t: str = None,
    stopping_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    **kwargs,
):
    """
    CHORDS-based parallel inference for Stable Diffusion (tiny model friendly).
    Removed all Flux-specific paths; uses UNet + CFG + scheduler.step.
    """
    device = self._execution_device
    batch_size = 1 if (prompt is None or isinstance(prompt, str)) else len(prompt)
    # Tiny 模型建议小分辨率快速验证
    height = height or 256
    width = width or 256

    # Encode text (with optional negative) and enable CFG if needed
    do_cfg = guidance_scale is not None and guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # Timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps  # tensor of shape [T]
    self._num_timesteps = len(timesteps)

    # Prepare initial latents
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        self.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # Solver for CHORDS: standard scheduler.step
    def solver(x_t, noise_pred, t_step, s_step):
        # t_step/s_step are indices; take actual timestep scalar
        t = timesteps[t_step]
        out = self.scheduler.step(noise_pred, t, x_t, return_dict=False)[0]
        return out.to(x_t.dtype)

    algorithm = CHORDS(
        T=len(timesteps),
        x0=latents,
        num_cores=num_cores,
        solver=solver,
        init_t=init_t,
        stopping_kwargs=stopping_kwargs,
        verbose=verbose,
    )

    # Timing (CUDA-safe & CPU-safe)
    use_cuda_timer = torch.cuda.is_available() and "cuda" in str(device)
    if use_cuda_timer:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
    else:
        import time
        _t0 = time.time()

    stats = {}
    pass_count = 0

    # Denoising loop with CHORDS allocation
    while True:
        allocation = algorithm.get_allocation()
        if not allocation:
            break

        computed_scores: Dict[int, torch.Tensor] = {}

        # Send all but the last slice to workers
        for thread_id, (t_idx, k, x_t) in enumerate(allocation[:-1]):
            mp_queues[0].put((
                x_t.to("cpu"),
                timesteps[t_idx].to("cpu"),
                prompt_embeds.to("cpu"),
                (negative_prompt_embeds.to("cpu") if (do_cfg and negative_prompt_embeds is not None) else None),
                float(guidance_scale) if do_cfg else 1.0,
                thread_id,
            ))

        # Compute the last slice locally
        t_idx, k, x_t = allocation[-1]
        t = timesteps[t_idx].to(x_t.dtype).to(device)

        if do_cfg and negative_prompt_embeds is not None:
            x_in = torch.cat([x_t, x_t], dim=0).to(device)
            e_in = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device)
            noise = self.unet(x_in, t, encoder_hidden_states=e_in).sample
            e_uncond, e_text = noise.chunk(2, dim=0)
            noise_pred = e_uncond + guidance_scale * (e_text - e_uncond)
        else:
            noise_pred = self.unet(x_t.to(device), t, encoder_hidden_states=prompt_embeds.to(device)).sample

        computed_scores[len(allocation) - 1] = noise_pred

        # Collect worker results
        for _ in range(len(allocation) - 1):
            noise_pred_cpu, thread_id = mp_queues[1].get()
            computed_scores[thread_id] = noise_pred_cpu.to(device)

        # Update CHORDS with all scores
        scores = []
        for thread_id, (t_idx, k, x_t) in enumerate(allocation):
            scores.append((t_idx, k, computed_scores[thread_id]))
        algorithm.update_scores(scores)
        algorithm.update_states(len(allocation))

        delete_ids, earlystop = algorithm.schedule_cores()
        if earlystop:
            break

        algorithm.cur_core_to_compute = algorithm.cur_core_to_compute[len(allocation):]
        if delete_ids:
            algorithm.cur_core_to_compute = [cid for cid in algorithm.cur_core_to_compute if cid not in delete_ids]

        pass_count += 1

    # Timing finalize
    if use_cuda_timer:
        end_evt.record()
        torch.cuda.synchronize()
        total_time_ms = float(start_evt.elapsed_time(end_evt))
    else:
        import time
        total_time_ms = (time.time() - _t0) * 1000.0

    # Retrieve final states
    hit_iter_idx, hit_x, hit_time = algorithm.get_last_x_and_hittime()

    stats["flops_count"] = algorithm.get_flops_count()
    stats["pass_count"] = pass_count
    stats["total_time_ms"] = total_time_ms
    for i in range(len(hit_x)):
        if i > 0:
            diff = torch.linalg.norm(hit_x[i] - hit_x[i - 1]).double().item() / hit_x[i].numel()
            stats[f"prev_diff_{hit_iter_idx[i]}"] = diff
        stats[f"hit_time_{hit_iter_idx[i]}"] = hit_time[i]

    # Decode
    if output_type == "latent":
        image = hit_x[-1]
    else:
        decode_seq = hit_x if full_return else [hit_x[-1]]
        images = []
        latents_list = []
        for z in decode_seq:
            latents_list.append(z.clone().cpu())
            z = z.to(device)
            # Stable Diffusion: scale only; no shift
            z = z / self.vae.config.scaling_factor
            img = self.vae.decode(z).sample
            img = self.image_processor.postprocess(img, output_type=output_type)
            images.append(img)
        image = (images, latents_list) if full_return else images[0]

    # Cleanup
    self.maybe_free_model_hooks()
    del algorithm
    torch.cuda.empty_cache()

    if not return_dict:
        return (image,)
    return StableDiffusionPipelineOutput(images=image), stats

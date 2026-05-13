import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class AsyncVAEVLMPipeline:
    """Runs per-chunk VAE decode + VLM submit on a dedicated background thread.

    Usage (per diffusion block):
        pipeline.submit(denoised_pred, chunk_key, ...)   # non-blocking
        ...                                               # main thread continues
        pixels = pipeline.get_pixels(chunk_key)          # blocks only if not ready
    """

    def __init__(self, vae, vlm_agent):
        self._vae = vae
        self._vlm_agent = vlm_agent

        # Single worker ensures causal VAE processes chunks in submission order.
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1)

        self._lock = threading.Lock()
        self._futures: Dict[str, Future] = {}
        self._pixel_store: Dict[str, torch.Tensor] = {}  # chunk_key -> CPU tensor
        self._chunk_order: List[str] = []
        self._vae_decode_times: List[float] = []  # ms, measured in background thread

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        denoised_pred: torch.Tensor,
        chunk_key: str,
        prompt_text: str,
        prompt_id: int,
        chunk_id: int,
        entities: list,
        global_registry: dict,
        is_first_chunk: bool,
    ) -> None:
        """Submit VAE decode + VLM scoring for one chunk. Non-blocking.

        denoised_pred is a GPU tensor produced by the main DiT stream.
        The background thread reads it after the main stream has finished
        writing (guaranteed by PyTorch's implicit stream ordering when the
        background thread calls .cpu(), which syncs all GPU work on that tensor).
        """
        with self._lock:
            self._chunk_order.append(chunk_key)

        future = self._executor.submit(
            self._run,
            denoised_pred,
            chunk_key,
            prompt_text,
            prompt_id,
            chunk_id,
            entities,
            global_registry,
            is_first_chunk,
        )
        with self._lock:
            self._futures[chunk_key] = future

    def get_pixels(
        self, chunk_key: str, timeout: float = 60.0
    ) -> Optional[torch.Tensor]:
        """Return decoded CPU pixel frames for chunk_key.

        Blocks only when the result is not yet available (should be rare given
        the 2-chunk lag for eviction, and the clean-context overlap for archival).
        """
        with self._lock:
            if chunk_key in self._pixel_store:
                return self._pixel_store[chunk_key]
            future = self._futures.get(chunk_key)

        if future is None:
            return None

        if not future.done():
            logger.warning(f"[AsyncVAEVLM] Main thread stalled waiting for {chunk_key}")

        try:
            future.result(timeout=timeout)
        except Exception as e:
            logger.error(f"[AsyncVAEVLM] VAE decode failed for {chunk_key}: {e}")
            return None

        with self._lock:
            return self._pixel_store.get(chunk_key)

    def get_ordered_chunks(self) -> List[torch.Tensor]:
        """Wait for all pending decodes and return pixel chunks in submission order."""
        with self._lock:
            keys = list(self._chunk_order)
            futures = [self._futures[k] for k in keys if k in self._futures]

        for f in futures:
            try:
                f.result()
            except Exception as e:
                logger.error(f"[AsyncVAEVLM] Error waiting for chunk future: {e}")

        with self._lock:
            return [self._pixel_store[k] for k in keys if k in self._pixel_store]

    def get_vae_decode_times(self) -> List[float]:
        """Return per-chunk VAE decode wall times (ms), in submission order."""
        with self._lock:
            return list(self._vae_decode_times)

    def clear(self) -> None:
        """Cancel pending futures and reset state (call before each inference run)."""
        with self._lock:
            futures = list(self._futures.values())
            self._futures.clear()
            self._pixel_store.clear()
            self._chunk_order.clear()
            self._vae_decode_times.clear()

        for f in futures:
            f.cancel()

    def shutdown(self) -> None:
        """Shutdown the worker thread pool. Call once at object teardown."""
        self.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run(
        self,
        denoised_pred: torch.Tensor,
        chunk_key: str,
        prompt_text: str,
        prompt_id: int,
        chunk_id: int,
        entities: list,
        global_registry: dict,
        is_first_chunk: bool,
    ) -> None:
        """VAE decode on background thread, then VLM submit.

        The background thread uses its own default CUDA stream. The .cpu()
        call at the end performs an implicit device synchronization, ensuring
        the decode is complete before we store the result or call VLM.
        """
        t0 = time.monotonic()

        with torch.no_grad():
            vae_param = next(self._vae.model.parameters(), None)
            if vae_param is not None:
                denoised_pred = denoised_pred.to(
                    device=vae_param.device, non_blocking=True
                )
            chunk_pixel = self._vae.decode_to_pixel_stream(denoised_pred)
            chunk_pixel = (chunk_pixel * 0.5 + 0.5).clamp(0, 1)

        # .cpu() syncs the background CUDA stream and transfers to CPU.
        chunk_pixel_cpu = chunk_pixel[0].cpu()

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        with self._lock:
            self._pixel_store[chunk_key] = chunk_pixel_cpu
            self._vae_decode_times.append(elapsed_ms)

        if self._vlm_agent is not None:
            self._vlm_agent.score_chunk_async(
                pixel_frames=chunk_pixel_cpu,
                prompt_text=prompt_text,
                prompt_id=prompt_id,
                chunk_id=chunk_id,
                entities=entities,
                global_registry=global_registry,
                is_first_chunk=is_first_chunk,
            )

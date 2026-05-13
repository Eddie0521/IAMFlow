from __future__ import annotations


def compute_pure_diffusion_time(
    diffusion_time: float,
    agent_time: float = 0.0,
    memory_time: float = 0.0,
    recache_time: float = 0.0,
    vlm_vae_time: float = 0.0,
) -> tuple[float, float]:
    """Compute pure denoising time by subtracting all overhead from diffusion loop."""
    pure_time = max(0.0, diffusion_time - agent_time - memory_time - recache_time - vlm_vae_time)
    pure_pct = 0.0 if diffusion_time == 0 else 100 * pure_time / diffusion_time
    return pure_time, pure_pct


def build_comparable_metrics(
    *,
    total_time: float,
    diffusion_time: float,
    final_output_time: float,
    pure_denoising_time: float,
    extra_output_decode_time: float = 0.0,
) -> dict[str, float]:
    """Build cross-method metrics with a consistent timing boundary.

    `diffusion_time` may include per-chunk output decode work (IAMFlow VLM mode).
    `extra_output_decode_time` captures that decode time so the returned
    `loop_no_output_decode_ms` is directly comparable to methods whose diffusion
    loop excludes output decode.
    """
    loop_no_output_decode_ms = max(0.0, diffusion_time - extra_output_decode_time)
    method_overhead_ms = max(0.0, loop_no_output_decode_ms - pure_denoising_time)
    output_decode_ms = max(0.0, extra_output_decode_time + final_output_time)
    return {
        "end_to_end_ms": total_time,
        "loop_no_output_decode_ms": loop_no_output_decode_ms,
        "core_dit_ms": max(0.0, pure_denoising_time),
        "method_overhead_ms": method_overhead_ms,
        "output_decode_ms": output_decode_ms,
    }


def summarize_memory_timings(memory_entries: list[dict[str, float | str]]) -> dict[str, float]:
    """Split IAMFlow memory timings into pure select/inject and VLM wait."""
    if not memory_entries:
        return {
            "total_ms": 0.0,
            "pure_ms": 0.0,
            "wait_ms": 0.0,
            "avg_total_ms": 0.0,
            "avg_pure_ms": 0.0,
            "avg_wait_ms": 0.0,
        }

    total_ms = sum(float(entry["total_ms"]) for entry in memory_entries)
    wait_ms = sum(float(entry.get("wait_ms", 0.0)) for entry in memory_entries)
    pure_ms = sum(
        max(0.0, float(entry["total_ms"]) - float(entry.get("wait_ms", 0.0)))
        for entry in memory_entries
    )
    count = len(memory_entries)
    return {
        "total_ms": total_ms,
        "pure_ms": pure_ms,
        "wait_ms": wait_ms,
        "avg_total_ms": total_ms / count,
        "avg_pure_ms": pure_ms / count,
        "avg_wait_ms": wait_ms / count,
    }

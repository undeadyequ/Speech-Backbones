from typing import Any, Dict, List, Optional, Tuple, Union
import torch

def convert_to_generator_input(
        model_name: str,
        x: Optional[torch.Tensor] = None,
        x_lengths: Optional[torch.Tensor] = None,
        n_timesteps: int = 50,
        temperature: float = 1.0,
        stoc: bool = False,
        spk: Optional[torch.Tensor] = None,
        length_scale: float = 0.91,
        emo_lab: Optional[torch.Tensor] = None,
        melstyle: Optional[torch.Tensor] = None,
        guidence_strength: float = 3.0,
):
    """
    Convert general input into args of specific models (Although currently only one) by converting
    parameter value etc.
    Args:
        model_name:
        text:
        style:

    Returns:

    """
    if model_name == "gradtts" or model_name == "gradtts_cross":
        generator_input = {
            "x": x,
            "x_lengths": x_lengths,
            "n_timesteps": n_timesteps,
            "temperature": temperature,
            "stoc": stoc,
            "spk": spk,
            "length_scale": length_scale,
            "emo_label": emo_lab,
            "melstyle": melstyle,
            "guidence_strength": guidence_strength,
        }
    else:
        generator_input = {
        }

    return generator_input



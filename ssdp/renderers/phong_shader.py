from typing import override

import jaxtyping as jt
import torch
import torch.nn as nn

from ssdp.utils.jaxtyping import jaxtyped


class PhongShader(nn.Module):
    @jaxtyped()
    @override
    def forward(
        self,
        ray_directions: jt.Float[torch.Tensor, " *B 3 "],
        surface_normals: jt.Float[torch.Tensor, " *B 3 "],
        light_directions: jt.Float[torch.Tensor, " *B 3 "],
        light_ambient_colors: jt.Float[torch.Tensor, " *B 3 "],
        light_diffuse_colors: jt.Float[torch.Tensor, " *B 3 "],
        light_specular_colors: jt.Float[torch.Tensor, " *B 3 "],
        material_ambient_colors: jt.Float[torch.Tensor, " *B 3 "],
        material_diffuse_colors: jt.Float[torch.Tensor, " *B 3 "],
        material_specular_colors: jt.Float[torch.Tensor, " *B 3 "],
        material_emission_colors: jt.Float[torch.Tensor, " *B 3 "],
        material_shininesses: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        ray_directions = nn.functional.normalize(ray_directions, p=2, dim=-1)
        surface_normals = nn.functional.normalize(surface_normals, p=2, dim=-1)
        light_directions = nn.functional.normalize(light_directions, p=2, dim=-1)

        cos_thetas = -torch.sum(light_directions * surface_normals, dim=-1, keepdim=True)
        diffuse_coefficients = nn.functional.relu(cos_thetas)

        reflected_directions = light_directions + 2.0 * surface_normals * cos_thetas

        cos_alphas = -torch.sum(reflected_directions * ray_directions, dim=-1, keepdim=True)
        specular_coefficients = nn.functional.relu(cos_alphas) ** material_shininesses

        ambient_colors = material_ambient_colors * light_ambient_colors
        diffuse_colors = material_diffuse_colors * light_diffuse_colors * diffuse_coefficients
        specular_colors = material_specular_colors * light_specular_colors * specular_coefficients
        colors = ambient_colors + diffuse_colors + specular_colors + material_emission_colors

        colors = torch.clamp(colors, min=0.0, max=1.0)

        return colors

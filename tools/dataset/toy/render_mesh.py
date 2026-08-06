from __future__ import annotations

import dataclasses
import enum
import itertools
import json
import math
import re
from pathlib import Path

import loguru
import matplotlib as mpl
import mitsuba as mi
import numpy as np
import scipy as sp
import torch
import tqdm
import trimesh
import tyro
from matplotlib import colors
from scipy.spatial.transform import Rotation

from nerfstudio.models.base_surface_model import SurfaceModel
from nerfstudio.utils.eval_utils import eval_setup
from ssdp.fields import SDF


@dataclasses.dataclass
class PoseConfig:
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target: tuple[float, float, float] = (0.0, 0.0, 1.0)
    upward: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def instantiate(self) -> mi.ScalarTransform4f:
        pose = mi.ScalarTransform4f().look_at(
            origin=self.origin,
            target=self.target,
            up=self.upward,
        )
        return pose


@dataclasses.dataclass
class TransformConfig:
    pose_config: PoseConfig = dataclasses.field(
        default_factory=PoseConfig,
    )
    scale_factors: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def instantiate(self) -> mi.ScalarTransform4f:
        pose = self.pose_config.instantiate()
        pose = pose.scale(self.scale_factors)
        return pose


class MaterialType(enum.StrEnum):
    DIFFUSE = enum.auto()
    DIELECTRIC = enum.auto()
    THINDIELECTRIC = enum.auto()
    ROUGHDIELECTRIC = enum.auto()
    CONDUCTOR = enum.auto()
    ROUGHCONDUCTOR = enum.auto()
    PLASTIC = enum.auto()
    ROUGHPLASTIC = enum.auto()
    NULL = enum.auto()


@dataclasses.dataclass
class MaterialConfig:
    material_type: MaterialType
    diffuse_reflectance: tuple[float, float, float] = (0.5, 0.5, 0.5)
    specular_roughness: float = 0.1
    interior_ior: str = "bk7"
    exterior_ior: str = "air"
    conductor_ior: str = "none"
    normal_distribution: str = "beckmann"
    sample_visible_normals: bool = True
    enable_internal_color_shifts: bool = False

    def instantiate(self) -> mi.BSDF:
        material = dict(type=self.material_type)

        if self.material_type is MaterialType.DIFFUSE:
            material.update(
                reflectance=dict(
                    type="rgb",
                    value=self.diffuse_reflectance,
                ),
            )

        elif (
            self.material_type is MaterialType.DIELECTRIC
            or self.material_type is MaterialType.THINDIELECTRIC
            or self.material_type is MaterialType.ROUGHDIELECTRIC
            or self.material_type is MaterialType.PLASTIC
            or self.material_type is MaterialType.ROUGHPLASTIC
        ):
            material.update(
                int_ior=self.interior_ior,
                ext_ior=self.exterior_ior,
            )

            if (
                self.material_type is MaterialType.PLASTIC
                or self.material_type is MaterialType.ROUGHPLASTIC
            ):
                material.update(
                    diffuse_reflectance=dict(
                        type="rgb",
                        value=self.diffuse_reflectance,
                    ),
                    nonlinear=self.enable_internal_color_shifts,
                )

            if (
                self.material_type is MaterialType.ROUGHDIELECTRIC
                or self.material_type is MaterialType.ROUGHPLASTIC
            ):
                material.update(
                    alpha=self.specular_roughness,
                    distribution=self.normal_distribution,
                    sample_visible=self.sample_visible_normals,
                )

        elif (
            self.material_type is MaterialType.CONDUCTOR
            or self.material_type is MaterialType.ROUGHCONDUCTOR
        ):
            material.update(
                material=self.conductor_ior,
            )

            if self.material_type is MaterialType.ROUGHCONDUCTOR:
                material.update(
                    alpha=self.specular_roughness,
                    distribution=self.normal_distribution,
                    sample_visible=self.sample_visible_normals,
                )

        material = mi.load_dict(material)

        return material


class ParticipatingMediaType(enum.StrEnum):
    HOMOGENEOUS = enum.auto()


@dataclasses.dataclass
class ParticipatingMediaConfig:
    participating_media_type: ParticipatingMediaType
    scattering_albedo: tuple[float, float, float] = (0.75, 0.75, 0.75)
    extinction_coefficient: tuple[float, float, float] = (1.0, 1.0, 1.0)
    scale_factor: float = 1.0
    sample_emitters: bool = True

    def instantiate(self) -> mi.Medium:
        participating_media = dict(
            type=self.participating_media_type,
            albedo=dict(
                type="rgb",
                value=self.scattering_albedo,
            ),
            sigma_t=dict(
                type="rgb",
                value=self.extinction_coefficient,
            ),
            scale=self.scale_factor,
            sample_emitters=self.sample_emitters,
        )

        participating_media = mi.load_dict(participating_media)

        return participating_media


class EmitterType(enum.StrEnum):
    AREA = enum.auto()
    POINT = enum.auto()
    CONSTANT = enum.auto()
    SUNSKY = enum.auto()
    SPOT = enum.auto()
    DIRECTIONALAREA = enum.auto()
    DIRECTIONAL = enum.auto()


@dataclasses.dataclass
class EmitterConfig:
    emitter_type: EmitterType
    radiometry: tuple[float, float, float] = (1.0, 1.0, 1.0)
    scale_factor: float = 1.0
    cutoff_angle: float = 30.0
    beam_width: float = 10.0
    sunsky_hour: float = 12.0
    pose_config: PoseConfig | None = None

    def instantiate(self) -> mi.Emitter:
        emitter = dict(type=self.emitter_type)

        if (
            self.emitter_type is EmitterType.AREA
            or self.emitter_type is EmitterType.CONSTANT
            or self.emitter_type is EmitterType.DIRECTIONALAREA
        ):
            radiance = np.multiply(self.radiometry, self.scale_factor)
            emitter.update(
                radiance=dict(
                    type="rgb",
                    value=radiance,
                ),
            )

        elif self.emitter_type is EmitterType.POINT or self.emitter_type is EmitterType.SPOT:
            intensity = np.multiply(self.radiometry, self.scale_factor)
            emitter.update(
                type=self.emitter_type,
                intensity=dict(
                    type="rgb",
                    value=intensity,
                ),
            )
            if self.emitter_type is EmitterType.SPOT:
                emitter.update(
                    cutoff_angle=self.cutoff_angle,
                    beam_width=self.beam_width,
                )

        elif self.emitter_type is EmitterType.SUNSKY:
            albedo = self.radiometry
            emitter.update(
                hour=self.sunsky_hour,
                albedo=dict(
                    type="rgb",
                    value=albedo,
                ),
            )

        elif self.emitter_type is EmitterType.DIRECTIONAL:
            irradiance = np.multiply(self.radiometry, self.scale_factor)
            emitter.update(
                irradiance=dict(
                    type="rgb",
                    value=irradiance,
                ),
            )

        if not (
            self.emitter_type is EmitterType.AREA
            or self.emitter_type is EmitterType.CONSTANT
            or self.emitter_type is EmitterType.DIRECTIONALAREA
        ):
            if self.pose_config:
                pose = self.pose_config.instantiate()
                emitter.update(to_world=pose)

        emitter = mi.load_dict(emitter)

        return emitter


class ShapeType(enum.StrEnum):
    PLY = enum.auto()
    CUBE = enum.auto()
    SPHERE = enum.auto()
    RECTANGLE = enum.auto()
    DISK = enum.auto()
    CYLINDER = enum.auto()
    SDFGRID = enum.auto()
    ELLIPSOIDS = enum.auto()
    ELLIPSOIDSMESH = enum.auto()


@dataclasses.dataclass
class PLYConfig:
    mesh_file: Path | None = None


@dataclasses.dataclass
class SDFConfig:
    config_file: Path | None = None
    object_aabb: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (-1.0, 1.0, 200),
        (-1.0, 1.0, 200),
        (-1.0, 1.0, 200),
    )
    chunk_size: int = 100**3

    @torch.no_grad()
    def instantiate(self) -> mi.TensorXf:
        _, pipeline, _, _ = eval_setup(self.config_file)
        model: SurfaceModel = pipeline.model
        field: SDF = model.field
        xyz_grid = torch.meshgrid(
            *itertools.starmap(torch.linspace, reversed(self.object_aabb)),
            indexing="ij",
        )
        xyz_grid = torch.stack(list(reversed(xyz_grid)), dim=-1)
        sdf_grid = torch.cat(
            [
                field.sdf(xyz.to("cuda")).cpu()
                for xyz in tqdm.tqdm(
                    iterable=torch.split(
                        tensor=xyz_grid.flatten(0, -2),
                        split_size_or_sections=self.chunk_size,
                        dim=0,
                    ),
                    colour=colors.to_hex("dodgerblue"),
                    desc="Evaluating the SDF grid...",
                )
            ],
            dim=0,
        ).unflatten(0, xyz_grid.shape[:-1])
        sdf_grid = mi.TensorXf(sdf_grid)
        return sdf_grid


@dataclasses.dataclass
class ShapeConfig:
    shape_type: ShapeType
    ply_config: PLYConfig | None = None
    sdf_config: SDFConfig | None = None
    transform_config: TransformConfig | None = None
    material_config: MaterialConfig | None = None
    interior_media_config: ParticipatingMediaConfig | None = None
    exterior_media_config: ParticipatingMediaConfig | None = None
    emitter_config: EmitterConfig | None = None

    def instantiate(self) -> mi.Shape:
        shape = dict(type=self.shape_type)

        if self.shape_type is ShapeType.PLY:
            shape.update(filename=str(self.ply_config.mesh_file))

        elif self.shape_type is ShapeType.SDFGRID:
            sdf_grid = self.sdf_config.instantiate()
            shape.update(grid=sdf_grid)

        elif (
            self.shape_type is ShapeType.ELLIPSOIDS or self.shape_type is ShapeType.ELLIPSOIDSMESH
        ):
            scale_factors = [1.0, 1.0, 1.0]
            translation_vector = [0.0, 0.0, 0.0]
            quaternion_vector = [1.0, 0.0, 0.0, 0.0]
            if self.transform_config:
                pose_config = self.transform_config.pose_config
                scale_factors = self.transform_config.scale_factors
                pose = pose_config.instantiate().matrix.numpy()
                rotation_matrix, translation_vector = np.split(pose[:3, ...], [3], axis=-1)
                quaternion_vector = Rotation.from_matrix(rotation_matrix).as_quat(
                    scalar_first=True
                )
            shape.update(
                scales=mi.TensorXf([scale_factors]),
                centers=mi.TensorXf([translation_vector]),
                quaternions=mi.TensorXf([quaternion_vector]),
                extent=1.0,
            )
            if self.shape_type is ShapeType.ELLIPSOIDSMESH:
                sphere = trimesh.creation.icosphere(4)
                sphere.export(filename := "/tmp/sphere.ply")
                shape.update(shell=dict(type="ply", filename=filename))

        if self.transform_config:
            if (
                self.shape_type is not ShapeType.ELLIPSOIDS
                and self.shape_type is not ShapeType.ELLIPSOIDSMESH
            ):
                pose = self.transform_config.instantiate()
                shape.update(to_world=pose)

        if self.material_config:
            material = self.material_config.instantiate()
            shape.update(material=material)

        if self.interior_media_config:
            interior_media = self.interior_media_config.instantiate()
            shape.update(interior=interior_media)

        if self.exterior_media_config:
            exterior_media = self.exterior_media_config.instantiate()
            shape.update(exterior=exterior_media)

        if self.emitter_config:
            emitter = self.emitter_config.instantiate()
            shape.update(emitter=emitter)

        shape = mi.load_dict(shape)

        return shape


class FilterType(enum.StrEnum):
    BOX = enum.auto()
    TENT = enum.auto()
    GAUSSIAN = enum.auto()
    MITCHELL = enum.auto()
    CATMULLROM = enum.auto()
    LANCZOS = enum.auto()


@dataclasses.dataclass
class FilterConfig:
    filter_type: FilterType
    radius: float = 1.0
    stddev: float = 0.5
    alpha: float = 1.0 / 3.0
    beta: float = 1.0 / 3.0
    lobes: int = 3

    def instantiate(self) -> mi.ReconstructionFilter:
        filter = dict(type=self.filter_type)

        if self.filter_type is FilterType.TENT:
            filter.update(radius=self.radius)

        elif self.filter_type is FilterType.GAUSSIAN:
            filter.update(stddev=self.stddev)

        elif self.filter_type is FilterType.MITCHELL:
            filter.update(A=self.alpha, B=self.beta)

        elif self.filter_type is FilterType.LANCZOS:
            filter.update(lobes=self.lobes)

        filter = mi.load_dict(filter)

        return filter


class FilmType(enum.StrEnum):
    HDRFILM = enum.auto()


@dataclasses.dataclass
class FilmConfig:
    film_type: FilmType
    image_size: tuple[int, int] = (768, 576)
    pixel_format: str = "rgba"
    filter_config: FilterConfig | None = None

    def instantiate(self) -> mi.Film:
        film = dict(
            type=self.film_type,
            width=self.image_size[0],
            height=self.image_size[1],
            pixel_format=self.pixel_format,
        )

        if self.filter_config:
            filter = self.filter_config.instantiate()
            film.update(rfilter=filter)

        film = mi.load_dict(film)

        return film


class SamplerType(enum.StrEnum):
    INDEPENDENT = enum.auto()
    STRATIFIED = enum.auto()
    MULTIJITTER = enum.auto()
    ORTHOGONAL = enum.auto()
    LDSAMPLER = enum.auto()


@dataclasses.dataclass
class SamplerConfig:
    sampler_type: SamplerType
    sample_count: int = 4
    jitter: bool = True
    strength: int = 2.0
    seed: int = 0

    def instantiate(self) -> mi.Sampler:
        sampler = dict(
            type=self.sampler_type,
            sample_count=self.sample_count,
            seed=self.seed,
        )

        if (
            self.sampler_type is SamplerType.STRATIFIED
            or self.sampler_type is SamplerType.MULTIJITTER
            or self.sampler_type is SamplerType.ORTHOGONAL
        ):
            sampler.update(jitter=self.jitter)

            if self.sampler_type is SamplerType.ORTHOGONAL:
                sampler.update(strength=self.strength)

        sampler = mi.load_dict(sampler)

        return sampler


class SensorType(enum.StrEnum):
    ORTHOGRAPHIC = enum.auto()
    PERSPECTIVE = enum.auto()
    THINLENS = enum.auto()


@dataclasses.dataclass
class SensorConfig:
    sensor_type: SensorType
    clip_range: tuple[float, float] = (0.01, 10000.0)
    fov_angle: float = 60.0
    fov_axis: str = "x"
    focus_distance: float = 0.0
    aperture_radius: float = 0.0
    pose_config: PoseConfig = dataclasses.field(
        default_factory=PoseConfig,
    )
    film_config: FilmConfig = dataclasses.field(
        default_factory=FilmConfig,
    )
    sampler_config: SamplerConfig = dataclasses.field(
        default_factory=SamplerConfig,
    )

    def instantiate(self) -> mi.Sensor:
        sensor = dict(
            type=self.sensor_type,
            near_clip=self.clip_range[0],
            far_clip=self.clip_range[1],
        )

        if self.sensor_type is SensorType.PERSPECTIVE or self.sensor_type is SensorType.THINLENS:
            sensor.update(
                fov=self.fov_angle,
                fov_axis=self.fov_axis,
            )

            if self.sensor_type is SensorType.THINLENS:
                sensor.update(
                    focus_distance=self.focus_distance,
                    aperture_radius=self.aperture_radius,
                )

        pose = self.pose_config.instantiate()
        sensor.update(to_world=pose)

        film = self.film_config.instantiate()
        sensor.update(film=film)

        sampler = self.sampler_config.instantiate()
        sensor.update(sampler=sampler)

        sensor = mi.load_dict(sensor)

        return sensor


class IntegratorType(enum.StrEnum):
    PATH = enum.auto()
    VOLPATH = enum.auto()
    VOLPATHMIS = enum.auto()


@dataclasses.dataclass
class IntegratorConfig:
    integrator_type: IntegratorType
    max_depth: int = -1
    rr_depth: int = 5
    hide_emitters: bool = False

    def instantiate(self) -> mi.Integrator:
        integrator = dict(
            type=self.integrator_type,
            max_depth=self.max_depth,
            rr_depth=self.rr_depth,
            hide_emitters=self.hide_emitters,
        )

        integrator = mi.load_dict(integrator)

        return integrator


@dataclasses.dataclass
class SceneConfig:
    shape_configs: dict[str, ShapeConfig]
    emitter_configs: dict[str, ShapeConfig | EmitterConfig]
    sensor_config: SensorConfig = dataclasses.field(
        default_factory=SensorConfig,
    )
    integrator_config: IntegratorConfig = dataclasses.field(
        default_factory=IntegratorConfig,
    )

    def instantiate(self) -> mi.Scene:
        scene = dict(type="scene")

        for shape_name, shape_config in self.shape_configs.items():
            shape = shape_config.instantiate()
            scene |= {shape_name: shape}

        for emitter_name, emitter_config in self.emitter_configs.items():
            emitter = emitter_config.instantiate()
            scene |= {emitter_name: emitter}

        sensor = self.sensor_config.instantiate()
        scene.update(sensor=sensor)

        integrator = self.integrator_config.instantiate()
        scene.update(integrator=integrator)

        scene = mi.load_dict(scene)

        return scene


@dataclasses.dataclass
class MeshRenderer:
    mesh_file: Path
    output_dir: Path

    scene_config: SceneConfig = dataclasses.field(
        default_factory=lambda: SceneConfig(
            shape_configs=dict(
                ground_slab=ShapeConfig(
                    shape_type=ShapeType.CUBE,
                    transform_config=TransformConfig(
                        pose_config=PoseConfig(
                            origin=(0.0, 0.0, -0.5),
                            target=(0.0, 0.0, 1.0),
                            upward=(0.0, 1.0, 0.0),
                        ),
                        scale_factors=(100.0, 100.0, 0.01),
                    ),
                    material_config=MaterialConfig(
                        material_type=MaterialType.ROUGHPLASTIC,
                        diffuse_reflectance=(1.0, 1.0, 1.0),
                        specular_roughness=0.1,
                        interior_ior="polypropylene",
                    ),
                ),
                left_slab=ShapeConfig(
                    shape_type=ShapeType.CUBE,
                    transform_config=TransformConfig(
                        pose_config=PoseConfig(
                            origin=(-0.5, 0.0, 0.0),
                            target=(-0.5, 0.0, 1.0),
                            upward=(0.0, 1.0, 0.0),
                        ),
                        scale_factors=(0.01, 0.5, 0.5),
                    ),
                    material_config=MaterialConfig(
                        material_type=MaterialType.PLASTIC,
                        diffuse_reflectance=(0.25, 0.5, 1.0),
                        interior_ior="polypropylene",
                    ),
                ),
                middle_slab=ShapeConfig(
                    shape_type=ShapeType.CUBE,
                    transform_config=TransformConfig(
                        pose_config=PoseConfig(
                            origin=(0.0, 0.0, 0.0),
                            target=(0.0, 0.0, 1.0),
                            upward=(0.0, 1.0, 0.0),
                        ),
                        scale_factors=(0.01, 0.5, 0.5),
                    ),
                    material_config=MaterialConfig(
                        material_type=MaterialType.PLASTIC,
                        diffuse_reflectance=(1.0, 0.75, 0.25),
                        interior_ior="polypropylene",
                    ),
                ),
                right_slab=ShapeConfig(
                    shape_type=ShapeType.CUBE,
                    transform_config=TransformConfig(
                        pose_config=PoseConfig(
                            origin=(0.5, 0.0, 0.0),
                            target=(0.5, 0.0, 1.0),
                            upward=(0.0, 1.0, 0.0),
                        ),
                        scale_factors=(0.01, 0.5, 0.5),
                    ),
                    material_config=MaterialConfig(
                        material_type=MaterialType.PLASTIC,
                        diffuse_reflectance=(0.25, 0.5, 1.0),
                        interior_ior="polypropylene",
                    ),
                ),
            ),
            emitter_configs=dict(
                env_light=EmitterConfig(
                    emitter_type=EmitterType.CONSTANT,
                    radiometry=(1.0, 1.0, 1.0),
                    scale_factor=0.25,
                ),
                key_light=ShapeConfig(
                    shape_type=ShapeType.RECTANGLE,
                    transform_config=TransformConfig(
                        pose_config=PoseConfig(
                            origin=(2.0, -2.0, 2.0),
                            target=(0.0, 0.0, 0.0),
                            upward=(0.0, 0.0, 1.0),
                        ),
                        scale_factors=(1.0, 1.0, 1.0),
                    ),
                    emitter_config=EmitterConfig(
                        emitter_type=EmitterType.AREA,
                        radiometry=(1.0, 0.75, 0.5),
                        scale_factor=10.0,
                    ),
                ),
                rim_light=ShapeConfig(
                    shape_type=ShapeType.RECTANGLE,
                    transform_config=TransformConfig(
                        pose_config=PoseConfig(
                            origin=(-2.0, 2.0, 2.0),
                            target=(0.0, 0.0, 0.0),
                            upward=(0.0, 0.0, 1.0),
                        ),
                        scale_factors=(1.0, 1.0, 1.0),
                    ),
                    emitter_config=EmitterConfig(
                        emitter_type=EmitterType.AREA,
                        radiometry=(0.5, 0.75, 1.0),
                        scale_factor=10.0,
                    ),
                ),
            ),
            sensor_config=SensorConfig(
                sensor_type=SensorType.PERSPECTIVE,
                fov_angle=30.0,
                fov_axis="x",
                pose_config=PoseConfig(
                    origin=(0.0, -3.0, 0.0),
                    target=(0.0, 0.0, 0.0),
                    upward=(0.0, 0.0, 1.0),
                ),
                film_config=FilmConfig(
                    film_type=FilmType.HDRFILM,
                    image_size=(1000, 1000),
                    filter_config=FilterConfig(
                        filter_type=FilterType.GAUSSIAN,
                    ),
                ),
                sampler_config=SamplerConfig(
                    sampler_type=SamplerType.LDSAMPLER,
                    sample_count=256,
                ),
            ),
            integrator_config=IntegratorConfig(
                integrator_type=IntegratorType.PATH,
            ),
        )
    )
    scene_aabb: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = (
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
    )

    export_mesh: bool = True
    export_meta: bool = True
    num_azimuth_views: int = 8
    num_elevation_views: int = 2
    azimuth_range: tuple[float, float] = (-math.pi, math.pi)
    elevation_range: tuple[float, float] = (0.0, math.pi / 4.0)
    exported_mesh_regex: str = r"^(?!.*light).*$"

    def __call__(self) -> None:
        mi.set_variant("cuda_ad_rgb")

        self.scene_config.shape_configs.update(
            object=dataclasses.replace(
                self.scene_config.shape_configs["object"],
                ply_config=dataclasses.replace(
                    self.scene_config.shape_configs["object"].ply_config,
                    mesh_file=self.mesh_file,
                ),
            ),
        )

        scene = self.scene_config.instantiate()

        radius = np.linalg.norm(self.scene_config.sensor_config.pose_config.origin)
        sensors = [
            dataclasses.replace(
                self.scene_config.sensor_config,
                pose_config=dataclasses.replace(
                    self.scene_config.sensor_config.pose_config,
                    origin=(
                        radius * math.cos(elevation) * math.cos(azimuth),
                        radius * math.cos(elevation) * math.sin(azimuth),
                        radius * math.sin(elevation),
                    ),
                ),
            ).instantiate()
            for elevation in np.linspace(
                *self.elevation_range,
                self.num_elevation_views + 2,
            )[1:-1]
            for azimuth in np.linspace(
                *self.azimuth_range,
                self.num_azimuth_views + 1,
            )[:-1]
        ]

        frames = []

        for index, sensor in enumerate(
            tqdm.tqdm(
                iterable=sensors,
                colour=mpl.colors.cnames["dodgerblue"],
                desc="Rendering views...",
            )
        ):
            image = mi.render(scene, sensor=sensor)

            output_file = self.output_dir / "images" / f"{index:03d}.png"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            mi.util.write_bitmap(str(output_file), image)

            if self.export_meta:
                extrinsic_matrix = sensor.m_to_world.matrix.numpy().squeeze(-1)
                extrinsic_matrix = extrinsic_matrix @ np.diag([-1.0, -1.0, 1.0, 1.0])

                [sensor] = scene.sensors()
                width, height = sensor.film().size()
                intrinsic_matrix = sensor.projection_transform().matrix.numpy().squeeze(-1)
                intrinsic_matrix = intrinsic_matrix @ np.diag([-1.0, -1.0, 1.0, 1.0])
                intrinsic_matrix = np.diag([width, height, 1.0, 1.0]) @ intrinsic_matrix
                intrinsic_matrix = sp.linalg.block_diag(intrinsic_matrix[:3, :3], np.ones((1, 1)))

                frame = dict(
                    rgb_path=str(output_file.relative_to(self.output_dir)),
                    camtoworld=extrinsic_matrix.tolist(),
                    intrinsics=intrinsic_matrix.tolist(),
                )
                frames.append(frame)

        if self.export_mesh:
            mesh_dir = self.output_dir / "meshes"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            for shape in scene.shapes():
                if re.search(self.exported_mesh_regex, shape.id()):
                    if not shape.is_mesh():
                        raise ValueError(f"{shape.id()} is not a mesh.")
                    shape.write_ply(str(mesh_dir / f"{shape.id()}.ply"))
            meshes = list(map(trimesh.load, mesh_dir.iterdir()))
            mesh = trimesh.util.concatenate(meshes)
            mesh_file = self.output_dir / "mesh.ply"
            mesh.export(mesh_file)

        if self.export_meta:
            aabb = tuple(zip(*self.scene_aabb, strict=True))
            meta_data = dict(
                camera_model="OPENCV",
                width=width,
                height=height,
                scene_box=dict(aabb=aabb),
                frames=frames,
            )

            meta_file = self.output_dir / "meta_data.json"
            with meta_file.open("w") as fp:
                json.dump(meta_data, fp, indent=4)

        loguru.logger.success("Finished!")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        MeshRenderer,
        config=(tyro.conf.AvoidSubcommands,),
    )()

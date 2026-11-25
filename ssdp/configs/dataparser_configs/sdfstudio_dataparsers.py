from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from ssdp.dataparsers import CustomSDFStudioDataParserConfig

custom_sdfstudio_dataparser = DataParserSpecification(
    config=CustomSDFStudioDataParserConfig(),
    description="Custom SDFStudio dataparser that supports loading foreground masks.",
)

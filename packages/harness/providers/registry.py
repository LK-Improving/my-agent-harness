"""构造即冻结的注册表；拒绝重复身份，不提供运行期替换入口。"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from harness.contracts.providers import ModelCatalog, ModelDescriptor, ModelTask, ProviderDefinition


class ProviderResolutionError(ValueError):
    """仅包含稳定错误码，调用方不得把原配置或凭据拼入异常正文。"""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, init=False)
class ProviderRegistry:
    _definitions: Mapping[str, ProviderDefinition]
    _catalogs: Mapping[str, ModelCatalog]

    def __init__(
        self, definitions: Iterable[ProviderDefinition], catalogs: Iterable[ModelCatalog]
    ) -> None:
        definition_map: dict[str, ProviderDefinition] = {}
        catalog_map: dict[str, ModelCatalog] = {}
        for definition in definitions:
            # 重验边界，拒绝通过未校验的model_copy更新破坏配置约束。
            definition = ProviderDefinition.model_validate_json(definition.model_dump_json())
            if definition.provider_id in definition_map:
                raise ProviderResolutionError("PROVIDER_ALREADY_REGISTERED")
            definition_map[definition.provider_id] = definition
        for catalog in catalogs:
            catalog = ModelCatalog.model_validate_json(catalog.model_dump_json())
            if catalog.provider_id in catalog_map:
                raise ProviderResolutionError("MODEL_CATALOG_ALREADY_REGISTERED")
            if catalog.provider_id not in definition_map:
                raise ProviderResolutionError("PROVIDER_NOT_REGISTERED")
            catalog_map[catalog.provider_id] = catalog
        if set(definition_map) != set(catalog_map):
            raise ProviderResolutionError("MODEL_CATALOG_MISSING")
        object.__setattr__(self, "_definitions", MappingProxyType(definition_map))
        object.__setattr__(self, "_catalogs", MappingProxyType(catalog_map))

    def get(self, provider_id: str) -> ProviderDefinition:
        try:
            return self._definitions[provider_id]
        except KeyError:
            raise ProviderResolutionError("PROVIDER_NOT_REGISTERED") from None

    def catalog(self, provider_id: str) -> ModelCatalog:
        self.get(provider_id)
        return self._catalogs[provider_id]

    def model(self, provider_id: str, model_id: str, task_type: ModelTask) -> ModelDescriptor:
        for model in self.catalog(provider_id).models:
            if model.model_id == model_id and model.task_type == task_type:
                return model
        raise ProviderResolutionError("MODEL_NOT_REGISTERED")

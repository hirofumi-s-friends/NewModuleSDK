from uuid import uuid4
from ruamel.yaml import ruamel
import json

from azureml.pipeline.core.module import Module, ModuleVersion
from azureml.pipeline.steps import ModuleStep
from azureml.core import Workspace, Datastore, Dataset
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Experiment, RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import AmlCompute

from azureml.studio.core.utils.strutils import to_snake_case
from azureml.studio.core.utils.column_selection import ColumnSelectionBuilder

USE_STRUCTURED_ARGUMENTS = 'USE_STRUCTURED_ARGUMENTS'
IGNORE_PARAMS = {
    'ServingEntry', 'Target', 'MLCComputeType', 'PrepareEnvironment',
    'Script', 'Framework', 'maxRunDurationSeconds', 'InterpreterPath',
    'UserManagedDependencies', 'CondaDependencies', 'DockerEnabled', 'BaseDockerImage',
    'GpuSupport', 'HistoryEnabled', 'Arguments'
}


class AttrDict(dict):

    def __init__(self, name, fields: list):
        super().__init__()
        self._name = name

        self._fields = set(fields)
        self._fields_mapping = {to_snake_case(field): field for field in fields}

    def __getattr__(self, item):
        item = self._fields_mapping.get(item, item)
        if item in self:
            return self[item]
        return self.__getattribute__(item)

    def __setattr__(self, key, value):
        if self.is_reserved_field(key):
            return super().__setattr__(key, value)
        self[key] = value

    def __setitem__(self, key, value):
        field = self._fields_mapping.get(key, key)
        if field not in self._fields:
            raise AttributeError(f"Can't set attribute {key}")
        super().__setitem__(field, value)

    @property
    def fields(self):
        return list(self._fields)

    def is_valid_field(self, key):
        return key in self._fields

    @staticmethod
    def is_reserved_field(key):
        return key[0] == '_'


class ModuleStepX:

    _aml_compute = 'AmlCompute'
    _provisioning_succeeded = 'Succeeded'

    def __init__(self, module: Module, workspace: Workspace = None, compute_target: AmlCompute = None):
        self.module = module
        self.workspace = workspace
        self.default_module_version = self.get_default_module_version()
        self.datastore = workspace.get_default_datastore()
        self.compute_target = compute_target

        print(f"Initializing module: {module.name}")
        self.inputs = self._get_inputs()
        self.outputs = self._get_outputs()
        self.params = self._get_params()

    @classmethod
    def get(cls, workspace, name, compute_target=None):
        prefix = 'azureml://'
        try:
            module = Module.get(workspace, name=prefix + name)
        except Exception:
            module = Module.get(workspace, name=name)
        return cls(module, workspace, compute_target=compute_target)

    def _get_inputs(self):
        print(f"Get inputs")
        names = []
        for item in self.default_module_version.interface.inputs:
            print(f"inputs.{to_snake_case(item.name)}")
            names.append(item.name)
        print()
        return AttrDict('inputs', names)

    def _get_params(self):
        print(f"Get params")
        params = {}
        for item in self.default_module_version.interface.parameters:
            if item.name in IGNORE_PARAMS:
                continue
            default_value = None if item.is_optional else getattr(item, 'default_value', None)
            default_value = f"'{default_value}'" if isinstance(default_value, str) else default_value
            params[item.name] = default_value
            print(f"params.{to_snake_case(item.name)} = {default_value}, is_optional={item.is_optional}")
        result = AttrDict('params', list(params.keys()))
        result.update(params)
        print()
        return result

    def _get_outputs(self):
        print(f"Get outputs")
        names = []
        for item in self.default_module_version.interface.outputs:
            names.append(item.name)
            print(f"outputs.{to_snake_case(item.name)}")
        outputs = AttrDict('outputs', names)
        for name in names:
            outputs[name] = PipelineData(uuid4().hex, datastore=self.datastore, is_directory=True)
        print()
        return outputs

    def get_module_step(self):
        print(f"ModuleStep {self.module.name}")
        print("Inputs: ", self.inputs)
        print("Outputs: ", self.outputs)
        print("Parameters: ", self.params)
        print("\n")

        params = {'Arguments': USE_STRUCTURED_ARGUMENTS}
        for key, val in self.params.items():
            if val is None:
                continue
            if isinstance(val, ColumnSelectionBuilder):
                val = json.dumps(val._obj)
            params[key] = val

        return ModuleStep(
            self.module,
            outputs_map=self.outputs,
            inputs_map=self.inputs,
            compute_target=self.get_compute_target(),
            params=params,
            runconfig=self.get_run_config(),
        )

    def get_compute_target(self):
        def compute_available(compute):
            return compute.type.upper() == self._aml_compute.upper() \
                   and compute.provisioning_state.upper() == self._provisioning_succeeded.upper()

        if self.compute_target is None:
            computes = AmlCompute.list(self.workspace)
            default_compute = next((compute for compute in computes if compute_available(compute)), None)
            if default_compute is None:
                raise EnvironmentError(f"No compute target available in workspace {self.workspace.name}!")
            return default_compute
        return self.compute_target

    def get_run_config(self):
        def _get_structured_interface_param(name, param_list):
            return next((param for param in param_list if param.name == name), None)

        param_list = self.default_module_version.interface.parameters
        conda_content = _get_structured_interface_param('CondaDependencies', param_list).default_value
        docker_enabled = _get_structured_interface_param('DockerEnabled', param_list).default_value
        base_docker_image = _get_structured_interface_param('BaseDockerImage', param_list).default_value
        conda_dependencies = CondaDependencies(_underlying_structure=ruamel.yaml.safe_load(conda_content))

        run_config = RunConfiguration()
        run_config.environment.docker.enabled = docker_enabled
        run_config.environment.docker.base_image = base_docker_image
        run_config.environment.python.conda_dependencies = conda_dependencies
        return run_config

    def get_default_module_version(self):
        for v in self.module.module_version_list():
            if v.version == self.module.default_version:
                return ModuleVersion.get(self.workspace, v.module_version_id)
        raise ModuleNotFoundError()


class DatasetX:
    DEFAULT_GLOBAL_DATASET_STORE = "azureml_globaldatasets"
    DEFAULT_DATA_REFERENCE_NAME = "Dataset"

    @classmethod
    def get(cls, workspace, name):
        dataset = Dataset.get_by_name(workspace, name)
        return dataset.as_named_input('dataset').as_mount('tmp/dataset')

    @classmethod
    def get_by_data_reference(cls, workspace, path):
        data_store = Datastore(workspace, cls.DEFAULT_GLOBAL_DATASET_STORE)
        return DataReference(
            datastore=data_store,
            data_reference_name=cls.DEFAULT_DATA_REFERENCE_NAME,
            path_on_datastore=path,
        )


class ExperimentX:
    @staticmethod
    def submit(workspace, steps, exp_name):
        pipeline = Pipeline(workspace, steps=[step.get_module_step() for step in steps])
        experiment = Experiment(workspace, exp_name)
        experiment.submit(pipeline)

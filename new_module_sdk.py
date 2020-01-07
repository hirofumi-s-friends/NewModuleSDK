from uuid import uuid4
from ruamel.yaml import ruamel

from azureml.pipeline.core.module import Module, ModuleVersion
from azureml.pipeline.steps import ModuleStep
from azureml.core import Workspace, Datastore, Dataset
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Experiment, RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import AmlCompute


USE_STRUCTURED_ARGUMENTS = 'USE_STRUCTURED_ARGUMENTS'


class AttrDict(dict):

    def __init__(self, name, fields: list):
        super().__init__()
        self._fields = set(fields)
        self._name = name
        print(f"AttrDict {name} is inited, fields={fields}")

    def __getattr__(self, item):
        if item in self:
            return self[item]
        return self.__getattribute__(item)

    def __setattr__(self, key, value):
        if self.is_reserved_field(key):
            return super().__setattr__(key, value)
        self[key] = value

    def __setitem__(self, key, value):
        if key not in self._fields:
            raise AttributeError(f"Can't set attribute {key}")
        super().__setitem__(key, value)

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
    _reserved_attr_names = ['inputs', 'outputs', 'params', 'inputs_keys', 'outputs_keys', 'params_keys']

    def __init__(self, module: Module, workspace: Workspace = None, compute_target: AmlCompute = None):
        self.module = module
        self.workspace = workspace
        self.default_module_version = self.get_default_module_version()

        interface_keys = self.get_interface_keys()
        self.inputs = AttrDict('Input', interface_keys['input'])
        self.outputs = AttrDict('Output', interface_keys['output'])
        self.params = AttrDict('Parameter', interface_keys['param'])

        self.datastore = workspace.get_default_datastore()
        self.compute_target = compute_target

        self.init_outputs()
        self.init_params()

    @classmethod
    def get(cls, workspace, name, compute_target=None):
        return cls(Module.get(workspace, name=name), workspace, compute_target=compute_target)

    def get_interface_keys(self):
        return {
            'input': [item.name for item in self.default_module_version.interface.inputs],
            'output': [item.name for item in self.default_module_version.interface.outputs],
            'param': [item.name for item in self.default_module_version.interface.parameters],
        }

    def init_outputs(self):
        for key in self.outputs.fields:
            self.outputs[key] = PipelineData(uuid4().hex, datastore=self.datastore, is_directory=True)

    def init_params(self):
        self.params['Arguments'] = USE_STRUCTURED_ARGUMENTS

    def get_module_step(self):
        print(f"ModuleStep {self.module.name}")
        print("Inputs: ", self.inputs)
        print("Outputs: ", self.outputs)
        print("Parameters: ", self.params)
        print("\n")

        return ModuleStep(
            self.module,
            outputs_map=self.outputs,
            inputs_map=self.inputs,
            compute_target=self.get_compute_target(),
            params=self.params,
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

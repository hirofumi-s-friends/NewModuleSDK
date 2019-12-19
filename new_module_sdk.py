from uuid import uuid4
from azureml.pipeline.core.module import Module
from azureml.pipeline.steps import ModuleStep
from azureml.core import Workspace, Datastore, Dataset
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Experiment, RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import AmlCompute


class NewModuleStep:

    def __init__(self, module: Module, ws: Workspace = None):
        self.module = module
        self.inputs = {}
        self.outputs = {}
        self.params = {}
        self.ws = ws
        self.ds = ws.get_default_datastore()

        self.init_outputs()
        self.init_params()

    @classmethod
    def get(cls, ws, name):
        return cls(Module.get(ws, name=name), ws)

    def init_outputs(self):
        # Should list output keys according to the module
        output_keys = ['Results_dataset']
        for key in output_keys:
            self.outputs[key] = PipelineData(uuid4().hex, datastore=self.ds, is_directory=True)

    def init_params(self):
        # Should list all params and set the default values
        self.params['Arguments'] = 'USE_STRUCTURED_ARGUMENTS'

    def __setattr__(self, key, value):
        if self.is_input_port(key):
            self.inputs[key] = value
        elif self.is_param(key):
            self.params[key] = value
        elif self.is_output_port(key):
            if value is not PipelineData:
                raise ValueError(f"Output must be set with PipelineData.")
        super().__setattr__(key, value)

    def __getattr__(self, item):
        for kv in [self.inputs, self.outputs, self.params]:
            if item in kv:
                return kv[item]
        return super().__getattribute__(item)

    def is_input_port(self, key):
        # Should check the key according to the module
        return key == 'Dataset'

    def is_output_port(self, key):
        # Should check the key according to the module
        return key == 'Results_dataset'

    def is_param(self, key):
        # Should check the key according to the module
        return False

    def get_module_step(self):
        print(self.inputs, self.outputs, self.params)
        return ModuleStep(
            self.module,
            outputs_map=self.outputs,
            inputs_map=self.inputs,
            compute_target=self.get_compute_target(),
            params=self.params,
            runconfig=self.get_run_config(),
        )

    def get_compute_target(self):
        # Should choose compute according to CPU/GPU of the module
        return AmlCompute(self.ws, 'test-compute-ds2')

    def get_run_config(self):
        # Should choose proper run_config according to the module
        run_config = RunConfiguration()
        conda_dependencies = CondaDependencies(_underlying_structure={
            'name': 'project_environment', 'channels': ['defaults'],
            'dependencies': [
                'python=3.6.8',
                {'pip': [
                    'azureml-dataprep[pandas,fuse]==1.1.29',
                    'azureml-designer-classic-modules==0.0.105'
                ]}]})
        run_config.environment.docker.enabled = True
        run_config.environment.docker.base_image = "mcr.microsoft.com/azureml/base:intelmpi2018.3-ubuntu16.04"
        run_config.environment.python.conda_dependencies = conda_dependencies
        return run_config


class NewDataset:

    @classmethod
    def get(cls, ws, name):
        # It would be better to check the dataset name instead of try
        try:
            dataset = Dataset.get_by_name(ws, name)
            return dataset.as_named_input('dataset').as_mount('tmp/dataset')
        except:
            # It would be better to get path according to the name
            global_dataset_datastore = Datastore(ws, "azureml_globaldatasets")
            return DataReference(
                datastore=global_dataset_datastore,
                data_reference_name='tmp_reference',
                path_on_datastore=name,
            )


def submit_experiment(ws, steps, exp_name):
    pipeline = Pipeline(ws, steps=[step.get_module_step() for step in steps])
    experiment = Experiment(ws, exp_name)
    experiment.submit(pipeline)

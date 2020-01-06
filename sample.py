from new_module_sdk import Workspace, DatasetX, ModuleStepX, ExperimentX


if __name__ == '__main__':
    ws = Workspace.from_config()
    dataset = DatasetX.get_by_data_reference(ws, "GenericCSV/Automobile_price_data_(Raw)")
    module_name = 'azureml://Convert to Dataset'
    step1 = ModuleStepX.get(ws, name=module_name)
    step1.inputs.Dataset = dataset
    step2 = ModuleStepX.get(ws, name=module_name)
    step2.inputs.Dataset = step1.outputs.Results_dataset
    ExperimentX.submit(ws, [step1, step2], 'test_experiment_ex')

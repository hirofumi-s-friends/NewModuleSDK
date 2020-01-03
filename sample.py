from new_module_sdk import Workspace, DatasetX, ModuleStepX, ExperimentX


if __name__ == '__main__':
    ws = Workspace.get(
        name='weda-test-ws-eus',
        resource_group='weda-test',
        subscription_id='e9b2ec51-5c94-4fa8-809a-dc1e695e4896'
    )
    dataset = DatasetX.get_by_data_reference(ws, "GenericCSV/Automobile_price_data_(Raw)")
    module_name = 'azureml://Convert to Dataset'
    step1 = ModuleStepX.get(ws, name=module_name)
    step1.Dataset = dataset
    step2 = ModuleStepX.get(ws, name=module_name)
    step2.Dataset = step1.Results_dataset
    ExperimentX.submit(ws, [step1, step2], 'test_experiment_ex')

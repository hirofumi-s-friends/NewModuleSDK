from new_module_sdk import Workspace, NewDataset, NewModuleStep, submit_experiment

if __name__ == '__main__':
    ws = Workspace.get(
        name='your-name',
        resource_group='your-rg',
        subscription_id='your-sub'
    )
    dataset = NewDataset.get(ws, "GenericCSV/Automobile_price_data_(Raw)")
    module_name = 'azureml://Convert to Dataset'
    step1 = NewModuleStep.get(ws, name=module_name)
    step1.Dataset = dataset
    step2 = NewModuleStep.get(ws, name=module_name)
    step2.Dataset = step1.Results_dataset
    submit_experiment(ws, [step1, step2], 'test_experiment')

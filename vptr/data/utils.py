import json

def get_task_id_mapping(task_folders, task_aliasing_dict=None, index=-3):
    task_descriptions = set()
    for task_folder in task_folders:
        index_shift = 'bridge_data_v2' in task_folder
        task_description = task_folder.split('/')[index - index_shift]
        if task_aliasing_dict and task_description in task_aliasing_dict:
            task_description = task_aliasing_dict[task_description]
        task_descriptions.add(task_description)
    task_descriptions = sorted(task_descriptions)
    task_dict = {task_descp: index for task_descp, index in
            zip(task_descriptions, range(len(task_descriptions)))}
    print ('Printing task descriptions..............')
    for idx, desc in task_dict.items():
        print (idx, ' : ', desc)
    print ('........................................')
    return task_dict

def load_task_id_mapping(restore_path, target_task_folders, task_aliasing_dict, placeholder_task=None, index=-3):
    checkpoint_dir = '/'.join(restore_path.split('/')[:-1])
    config_path = f'{checkpoint_dir}/config.json'
    print(f'Restoring task id mapping from {config_path}.')
    with open(config_path, 'r') as f:
        config = json.load(f)
    task_id_mapping = config['task_id_mapping']
    for k, v in list(task_id_mapping.items()):
        task_id_mapping[task_aliasing_dict.get(k, k)] = task_id_mapping.pop(k)

    target_descriptions = []
    for task_folder in target_task_folders:
        index_shift = 'bridge_data_v2' in task_folder
        task_description = task_folder.split('/')[index - index_shift]
        target_descriptions.append(task_aliasing_dict.get(task_description, task_description))
    assert len(set(target_descriptions)) == 1

    target_description = target_descriptions[0]
    if target_description not in task_id_mapping:
        assert len(placeholder_task) > 0
        task_id_mapping[target_description] = task_id_mapping.pop(placeholder_task)

    return task_id_mapping

def exclude_tasks(paths, excluded_tasks):
    new_paths = []
    for d in paths:
        reject = False
        for exdir in excluded_tasks:
            if exdir in d:
                # print('excluding', d)
                reject = True
                break
        if not reject:
            new_paths.append(d)
    return new_paths

def include_tasks(paths, included_tasks):
    new_paths = []
    for d in paths:
        accept = False
        for exdir in included_tasks:
            if exdir in d:
                accept = True
                break
        if accept:
            new_paths.append(d)
    return new_paths

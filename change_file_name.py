from pathlib import Path
import yaml

log_file = '/scratch/bcqn/ywu20/odin_processed/train_validation_database.yaml'

with open(log_file, 'r') as f:
    data = yaml.safe_load(f)

new_data = data.copy()

for d in new_data:
    d['filepath'] = d['filepath'].replace('/mnt/data', '/scratch/bcqn/ywu20')
    d['instance_gt_filepath'] = d['instance_gt_filepath'].replace('/mnt/data', '/scratch/bcqn/ywu20')


# Save the new data as yaml file

with open(log_file, 'w') as f:
    yaml.dump(new_data, f)

print('Done!')
import argparse
import torch
import json
from safetensors.torch import save_file
def convert_to_safetensors(pth_file_path: str, output_filename: str):
  """
  Converts a PyTorch .pth file to a .safetensors file,
  saving the 'config' separately.
  """
  try:
    state_dict = torch.load(pth_file_path, map_location=torch.device('cpu'))

    # Extract config 
    config = state_dict.pop('config', None)
    if config:
        config_filename = output_filename + '.config.json'
        with open(config_filename, 'w') as f:
            json.dump(config, f)
        print(f"Saved config to '{config_filename}'")


    # Flatten the state_dict, excluding non-tensor values
    flattened_state_dict = {}
    def flatten(d, parent_key=''):
      for k, v in d.items():
        new_key = parent_key + '_' + k if parent_key else k
        if isinstance(v, dict):
          flatten(v, new_key)
        elif isinstance(v, torch.Tensor): # Save only tensors
          flattened_state_dict[new_key] = v

    flatten(state_dict)

    # Save the flattened tensors to a safetensors file
    save_file(flattened_state_dict, output_filename)
    print(f"Successfully converted '{pth_file_path}' to '{output_filename}'")

  except FileNotFoundError:
    print(f"Error: File not found: {pth_file_path}")
  except Exception as e:
    print(f"Error converting '{pth_file_path}': {e}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Convert a PyTorch .pth file to .safetensors.')
  parser.add_argument('pth_file', help='Path to the .pth file.')
  parser.add_argument('output_file', help='Desired filename for the output .safetensors file.')
  args = parser.parse_args()

  convert_to_safetensors(args.pth_file, args.output_file)
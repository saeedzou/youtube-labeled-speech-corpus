import argparse
import os
import random

def split_words(txt_path, num_splits, output_dir):
    """
    Reads a text file, shuffles its lines, and splits them into multiple files.

    Args:
        txt_path (str): Path to the input text file (one phrase per line).
        num_splits (int): The number of files to split the lines into.
        output_dir (str): The directory to save the output files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read lines from the input file
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Shuffle the lines
    random.shuffle(lines)

    # Determine the size of each split
    split_size = len(lines) // num_splits
    extra = len(lines) % num_splits

    # Split the lines and write to output files
    start_index = 0
    for i in range(num_splits):
        end_index = start_index + split_size + (1 if i < extra else 0)
        split_lines = lines[start_index:end_index]
        
        output_filename = os.path.join(output_dir, f'split_{i+1:04d}.txt')
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.writelines(split_lines)
            
        start_index = end_index

    print(f"Successfully split '{txt_path}' into {num_splits} files in '{output_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shuffle and split a text file into multiple parts.')
    parser.add_argument('--txt_path', type=str, required=True, help='Path to the input text file (one phrase per line).')
    parser.add_argument('--num_splits', type=int, required=True, help='The number of files to split the lines into.')
    parser.add_argument('--output_dir', type=str, required=True, help='The directory to save the output files.')

    args = parser.parse_args()

    split_words(args.txt_path, args.num_splits, args.output_dir)

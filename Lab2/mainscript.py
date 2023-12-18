import argparse
from annotation import create_annotation_file
from dataset import copy_dataset
from iterator import MultiClassDatasetIterator


def main():
    parser = argparse.ArgumentParser(description='Dataset manipulation script')
    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand')

    copy_parser = subparsers.add_parser('dataset', help='Copy and rename or randomize dataset')
    copy_parser.add_argument('src_folder', type=str, help='Source folder path')
    copy_parser.add_argument('dest_folder', type=str, help='Destination folder path')
    copy_parser.add_argument('--randomize', action='store_true', help='Assign random numbers?')
    copy_parser.set_defaults(func=copy_dataset)

    create_parser = subparsers.add_parser('annotation', help='Create annotation file')
    create_parser.add_argument('folder_path', type=str, help='Main folder path')
    create_parser.add_argument('subfolder_paths', nargs='+', type=str, help='List of subfolder paths')
    create_parser.add_argument('annotation_file_path', type=str, help='Path for the annotation file')
    create_parser.set_defaults(func=create_annotation_file)

    iterate_parser = subparsers.add_parser('iterator', help='Iterate over annotation')
    iterate_parser.add_argument('annotation_file', type=str, help='Path to annotation')
    iterate_parser.add_argument('class_names', nargs='+', type=str, help='List of class names')
    iterate_parser.set_defaults(func=iterator)

    args = parser.parse_args()
    args.func(args)

def iterator(args):
    annotation_file = args.annotation_file
    class_names = args.class_names

    multi_class_iterator = MultiClassDatasetIterator(annotation_file, class_names)

    for instance in multi_class_iterator:
        print(instance)

if __name__ == '__main__':
    main()
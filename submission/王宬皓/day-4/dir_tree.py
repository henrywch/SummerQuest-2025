import os
import argparse

def list_dir(start_path, max_depth=None, current_depth=0, follow_symlinks=False):
    """Recursively list directories with depth control and symlink handling."""
    if current_depth == 0:
        start_path = os.path.abspath(start_path)
        print(start_path)

    if max_depth is not None and current_depth >= max_depth:
        return

    try:
        with os.scandir(start_path) as entries:
            for entry in entries:
                # Print entry with depth-based indentation
                print('  ' * (current_depth + 1) + entry.name)
                
                # Handle directory recursion
                if entry.is_dir(follow_symlinks=follow_symlinks):
                    list_dir(entry.path, max_depth, current_depth + 1, follow_symlinks)
                    
    except PermissionError:
        print('  ' * (current_depth + 1) + "[Permission Denied]")
    except FileNotFoundError:
        print('  ' * (current_depth + 1) + "[Directory Not Found]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recursive directory lister with depth control')
    parser.add_argument('directory', nargs='?', default='.', 
                        help='Starting directory (default: current directory)')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Maximum recursion depth (0=root only)')
    parser.add_argument('--follow-symlinks', action='store_true',
                        help='Follow symbolic links (caution: may cause cycles)')
    
    args = parser.parse_args()
    
    list_dir(
        args.directory,
        max_depth=args.max_depth,
        follow_symlinks=args.follow_symlinks
    )
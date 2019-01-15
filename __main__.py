import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Dict

import click as click
import numpy as np

VALID_RESULTS = {
    'img_000.jpg': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    'img_001.jpg': np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
    'img_002.jpg': np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
    'img_003.jpg': np.array([1, 0, 0, 1, 1, 0, 2, 2, 2, 0, 1, 1, 0, 1, 1]),
    'img_004.jpg': np.array([4, 4, 2, 7, 5, 5, 2, 2, 2, 0, 1, 1, 0, 1, 1]),
    'img_005.jpg': np.array([4, 4, 2, 7, 5, 5, 2, 2, 2, 0, 1, 1, 3, 3, 3]),
    'img_006.jpg': np.array([4, 4, 2, 7, 5, 5, 2, 2, 2, 0, 1, 1, 3, 3, 3]),
    'img_007.jpg': np.array([5, 5, 6, 4, 4, 5, 2, 1, 3, 3, 2, 2, 0, 0, 0]),
    'img_008.jpg': np.array([5, 5, 6, 4, 4, 5, 2, 3, 3, 5, 3, 5, 1, 0, 0]),
    'img_009.jpg': np.array([1, 7, 6, 9, 8, 8, 3, 3, 3, 5, 3, 5, 3, 3, 3]),
}


@click.command()
@click.argument('applications_directory', type=click.Path(exists=True, file_okay=False))
@click.argument('images_directory', type=click.Path(exists=True, file_okay=False))
@click.argument('output_directory', type=click.Path(exists=True, file_okay=False))
@click.option('--no-run', is_flag=True)
@click.option('--no-compute', is_flag=True)
def main(applications_directory: str, images_directory: str, output_directory: str, no_run: bool, no_compute: bool):
    applications_directory = Path(applications_directory)
    images_directory = Path(images_directory)
    output_directory = Path(output_directory)

    if not no_run:
        states = run_applications(applications_directory, images_directory, output_directory)
        with open(output_directory / 'states.json', 'w') as states_file:
            json.dump(states, states_file)

    if not no_compute:
        results = compute_results(output_directory)
        with open(output_directory / 'results.json', 'w') as results_file:
            json.dump(results, results_file)


def run_applications(applications_directory: Path, images_directory: Path, output_directory: Path) -> Dict[str, str]:
    states = {}
    for applications_directory_entry in applications_directory.iterdir():
        if applications_directory_entry.is_dir():
            student_name, status = process_application_directory(applications_directory_entry, images_directory,
                                                                 output_directory)
            states[student_name] = status

    return states


def compute_results(output_directory: Path) -> Dict[str, float]:
    results = {}
    for student_output_directory in output_directory.iterdir():
        results_file_path: Path = student_output_directory / 'results.txt'
        if results_file_path.exists():
            try:
                images_scores_sum = 0.0
                with open(results_file_path) as results_file:
                    for line in results_file:
                        split_line = line.split(' ')
                        ground_truth = VALID_RESULTS[split_line[0]]
                        image_results = np.array(split_line[1:], dtype=np.int)
                        images_scores_sum += np.sum(np.abs(image_results - ground_truth)) / np.sum(ground_truth)

                results[student_output_directory.name] = images_scores_sum / 15
            except Exception as e:
                print(f'{student_output_directory.name} failed: {e}', file=sys.stderr)

    return results


def process_application_directory(path: Path, images_directory: Path, output_dir: Path) -> Tuple[str, str]:
    for application_file in path.iterdir():
        if application_file.name.endswith('.exe'):
            student_name = application_file.name[:-4]
            student_output_dir = output_dir / student_name
            student_output_dir.mkdir(exist_ok=True)
            results_file = student_output_dir / 'results.txt'
            stdout_file = student_output_dir / 'stdout'
            stderr_file = student_output_dir / 'stderr'

            print(f'Running {student_name}...')
            try:
                with open(stdout_file, 'w') as stdout, open(stderr_file, 'w') as stderr:
                    subprocess.run([str(application_file), str(images_directory), str(results_file)],
                                   stdout=stdout, stderr=stderr, timeout=100)
            except subprocess.TimeoutExpired:
                return student_name, 'TIMEOUT'
            except subprocess.SubprocessError:
                return student_name, 'ERROR'

            return student_name, 'OK'

    return path.name, 'NOEXE'


if __name__ == '__main__':
    main()

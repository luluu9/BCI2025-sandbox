import os
import mne
from moabb.benchmark import benchmark
from moabb.datasets import PhysionetMI
from brainbot_dataset import get_brainbot_dataset
from datasets import PhysionetMI16, Weibo2014_16, Weibo2014_64_5_classes

def get_brainbot_datasets(subjects=10, max_trials=4, brainbot_intervals=[[0.0, 2.5]], events=None, code_suffix=""):
    brainbot_datasets = []
    for interval in brainbot_intervals:
        bb_ds = get_brainbot_dataset(interval=interval, events=events, code_suffix=code_suffix)
        bb_ds.n_sessions = min(max_trials, bb_ds.n_sessions)
        bb_ds.subject_list = bb_ds.subject_list[:subjects]
        brainbot_datasets.append(bb_ds)
    return brainbot_datasets

def get_all_datasets(subjects=10, max_trials=4, brainbot_intervals=[[0, 2.5], [0, 5]]):
    brainbot_datasets = get_brainbot_datasets(subjects=subjects, max_trials=max_trials, brainbot_intervals=brainbot_intervals)
    
    physionet_dataset = PhysionetMI()
    physionet_dataset.subject_list = physionet_dataset.subject_list[:subjects]
    
    physionet16_dataset = PhysionetMI16()
    physionet16_dataset.subject_list = physionet16_dataset.subject_list[:subjects]
    
    weibo2014_dataset = Weibo2014_64_5_classes()
    weibo2014_dataset.subject_list = weibo2014_dataset.subject_list[:subjects]
    
    weibo2014_16_dataset = Weibo2014_16()
    weibo2014_16_dataset.subject_list = weibo2014_16_dataset.subject_list[:subjects]
    
    assert len(weibo2014_16_dataset.subject_list) == subjects
    assert len(weibo2014_dataset.subject_list) == subjects
    assert len(physionet16_dataset.subject_list) == subjects
    assert len(physionet_dataset.subject_list) == subjects

    datasets = brainbot_datasets + [physionet16_dataset, physionet_dataset, weibo2014_dataset, weibo2014_16_dataset]
    return datasets

def print_results_summary(results_df):
    print("Results Summary:")
    summary = results_df.groupby(['pipeline', 'dataset'])['score'].agg(['mean', 'std', 'count'])
    summary['mean'] = summary['mean'].round(3)
    summary['std'] = summary['std'].round(3)
    print(summary.to_string())
    print("=" * 50)

    print("\nDetailed Results by Subject and Dataset:")
    detailed = results_df.pivot_table(
        index=['dataset', 'subject', 'session'],
        columns='pipeline',
        values='score'
    )
    print(detailed.round(3).to_string())
    print("=" * 50)


def run_moabb_benchmark(pipelines_dir, base_dir="./benchmarks", datasets_list=None, n_jobs=1, overwrite=False):
    pipelines_path = os.path.join(os.getcwd(), pipelines_dir)
    print(pipelines_path)

    if datasets_list is None:
        datasets_list = get_all_datasets()

    cache_config = dict(
        use=True,
        save_raw=True,
        save_epochs=True,
        save_array=True,
        overwrite_raw=False,
        overwrite_epochs=False,
        overwrite_array=False,
    )

    print("Using cache dir:", mne.get_config('MNE_DATA'))

    return benchmark(
        pipelines=pipelines_path,
        evaluations=["WithinSession"],
        paradigms=["MotorImagery"],
        include_datasets=datasets_list,
        results=os.path.join(base_dir, f"results-{pipelines_dir}"),
        overwrite=overwrite,
        plot=False,
        output=os.path.join(base_dir, f"output-{pipelines_dir}"),
        n_jobs=n_jobs,
        cache_config=cache_config
    )
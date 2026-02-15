from moabb.datasets import PhysionetMI
from brainbot_dataset import get_brainbot_dataset
from datasets import PhysionetMI16, Weibo2014_16, Weibo2014_64_5_classes


def get_all_datasets(subjects=10, max_trials=4):
    brainbot_dataset = get_brainbot_dataset()
    brainbot_dataset.n_sessions = min(max_trials, brainbot_dataset.n_sessions)
    brainbot_dataset.subject_list = brainbot_dataset.subject_list[:subjects]
    
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

    datasets = [brainbot_dataset, physionet16_dataset, physionet_dataset, weibo2014_dataset, weibo2014_16_dataset]
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


def run_moabb_benchmark(pipelines_dir, base_dir="./benchmarks", datasets_list=datasets, n_jobs=1):
    pipelines_path = os.path.join(os.getcwd(), pipelines_dir)
    print(pipelines_path)

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
        overwrite=False,
        plot=False,
        output=os.path.join(base_dir, f"output-{pipelines_dir}"),
        n_jobs=n_jobs,
        cache_config=cache_config
    )
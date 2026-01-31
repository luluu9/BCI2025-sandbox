from moabb.datasets import PhysionetMI, Weibo2014
from brainbot_dataset import get_brainbot_dataset
from datasets16 import PhysionetMI16, Weibo2014_16


def get_all_datasets(subjects=10, max_trials=4):
    brainbot_dataset = get_brainbot_dataset()
    brainbot_dataset.n_sessions = min(max_trials, brainbot_dataset.n_sessions)
    brainbot_dataset.subject_list = brainbot_dataset.subject_list[:subjects]
    
    physionet_dataset = PhysionetMI()
    physionet_dataset.subject_list = physionet_dataset.subject_list[:subjects]
    
    physionet16_dataset = PhysionetMI16()
    physionet16_dataset.subject_list = physionet16_dataset.subject_list[:subjects]
    
    weibo2014_dataset = Weibo2014()
    weibo2014_dataset.subject_list = weibo2014_dataset.subject_list[:subjects]
    
    weibo2014_16_dataset = Weibo2014_16()
    weibo2014_16_dataset.subject_list = weibo2014_16_dataset.subject_list[:subjects]
    
    assert len(weibo2014_16_dataset.subject_list) == subjects
    assert len(weibo2014_dataset.subject_list) == subjects
    assert len(physionet16_dataset.subject_list) == subjects
    assert len(physionet_dataset.subject_list) == subjects

    datasets = [brainbot_dataset, physionet16_dataset, physionet_dataset, weibo2014_dataset, weibo2014_16_dataset]
    return datasets

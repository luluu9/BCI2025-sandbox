
from moabb.analysis.results import Results
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery

def print_results_summary(results_df):
    print("Results Summary:")
    summary = results_df.groupby(['pipeline', 'dataset'])['score'].agg(['mean', 'std', 'count'])
    summary['mean'] = summary['mean'].round(3)
    summary['std'] = summary['std'].round(3)
    print(summary.to_string())
    print("=" * 50)

    print("\\nDetailed Results by Subject and Dataset:")
    detailed = results_df.pivot_table(
        index=['dataset', 'subject', 'session'],
        columns='pipeline',
        values='score'
    )
    print(detailed.round(3).to_string())
    print("=" * 50)

results = Results(  
    evaluation_class=WithinSessionEvaluation,  
    paradigm_class=MotorImagery,  
    hdf5_path="./benchmarks/results-pipelines_MI"
)

df = results.to_dataframe()  
print_results_summary(df)


results = Results(  
    evaluation_class=WithinSessionEvaluation,  
    paradigm_class=MotorImagery,  
    hdf5_path="./benchmarks/results-pipelines_MI_tensorflow"
)

df = results.to_dataframe()  
print_results_summary(df)

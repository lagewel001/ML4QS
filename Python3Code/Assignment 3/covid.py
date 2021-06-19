import copy
import numpy as np
import pandas as pd
import os
import scipy.interpolate as interp
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from Chapter4.TemporalAbstraction import NumericalAbstraction, CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter5.Clustering import NonHierarchicalClustering

OutlierDistr = DistributionBasedOutlierDetection()
MisVal = ImputationMissingValues()
LowPass = LowPassFilter()
PCA = PrincipalComponentAnalysis()
NumAbs = NumericalAbstraction()
FreqAbs = FourierTransformation()
CatAbs = CategoricalAbstraction()
clusteringNH = NonHierarchicalClustering()

# Load in and combine required data
hr_data = pd.read_csv('./data/heart_rate.csv')
hr_data.datetime = pd.to_datetime(hr_data.datetime)
hr_data = hr_data.sort_values('datetime').drop('is_resting', axis=1)

bp_data = pd.read_csv('./data/blood_pressure.csv')
bp_data.rename(columns={'measurement_datetime': 'datetime'}, inplace=True)
bp_data.datetime = pd.to_datetime(bp_data.datetime)
bp_data = bp_data.sort_values('datetime')

survey_data = pd.read_csv('./data/surveys.csv')
survey_data = survey_data[survey_data.scale == 'S_COVID_OVERALL'][['created_at', 'user_code', 'value']]
survey_data.rename(columns={'created_at': 'datetime', 'value': 'covid_symptoms_score'}, inplace=True)
survey_data.datetime = pd.to_datetime(survey_data.datetime)
survey_data = survey_data.sort_values('datetime')

hrv_data = pd.read_csv('./data/hrv_measurements.csv')
hrv_data.rename(columns={'measurement_datetime': 'datetime'}, inplace=True)
hrv_data.datetime = pd.to_datetime(hrv_data.datetime)
hrv_data = hrv_data.sort_values('datetime')
hrv_data = hrv_data[['user_code', 'datetime', 'meanrr', 'mxdmn', 'sdnn', 'rmssd', 'pnn50',
                     'mode', 'amo', 'lf', 'hf', 'vlf', 'lfhf', 'total_power']]

combi_data = pd.merge_asof(hr_data, survey_data, on='datetime', by='user_code')
combi_data = combi_data.dropna(subset=['covid_symptoms_score'])

combi_data = pd.merge_asof(combi_data, bp_data, on='datetime', by='user_code', direction='nearest')
combi_data = pd.merge_asof(combi_data, hrv_data, on='datetime', by='user_code')

combi_data.set_index('datetime', inplace=True)


def plot_covid_data(data, cols, labels, user_code, file_name):
    fig, axs = plt.subplots(2, figsize=(8, 5), sharex='all', gridspec_kw={'height_ratios': [5, 1]})

    for col, label in zip(cols, labels):
        axs[0].plot(data.index, data[col], label=label)
    axs[0].set_title('Measurements', loc='left')

    axs[1].plot(data.index, user_data.covid_symptoms_score, color='#d62728', label='covid score')
    axs[1].set_ylim((-1, 7))
    axs[1].set_title('Covid symptoms score', loc='left')

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(l, []) for l in zip(*lines_labels)]
    axs[1].legend(lines, labels, loc='upper center', bbox_to_anchor=(.5, -.65),
                  ncol=4, borderaxespad=0)

    fig.suptitle(f"User {user_code}")

    plt.savefig(file_name, bbox_inches='tight')
    plt.show()


granularity = 1
delta = 500
epsilon = delta / (1 + granularity)
for user_code, user_data in combi_data.groupby('user_code'):
    if len(user_data) > 99 and len(np.unique(user_data.covid_symptoms_score)) > 2:
        user_data = user_data.copy()
        dt_range_index = pd.date_range(user_data.index.min(), periods=len(user_data), freq=f"{delta}ms")
        user_data.index = dt_range_index

        # Plot original data
        plot_covid_data(user_data, ['heart_rate', 'diastolic', 'systolic'],
                        ['heart rate', 'bp diastolic', 'bp systolic'], user_code,
                        f"./user_data/blood_pressure_{user_code}_orig.png")

        # Remove duplicated blood pressure data caused by asof merge
        bp_cols = ['diastolic', 'systolic', 'functional_changes_index', 'circulatory_efficiency',
                   'kerdo_vegetation_index', 'robinson_index']
        user_data.loc[user_data.diastolic.diff(-1).fillna(user_data.diastolic) == 0, bp_cols] = np.nan

        # Increase data granularity based on heart rate
        intermediate_row = pd.Series(np.nan, user_data.columns)
        row_func = lambda d: d.append(intermediate_row, ignore_index=True)

        grp = np.arange(len(user_data)) // granularity
        user_data = user_data.groupby(grp, group_keys=False).apply(row_func).reset_index(drop=True)
        user_data.index = pd.date_range(dt_range_index.min(), periods=len(user_data), freq=f"{epsilon}ms")
        user_data.user_code = user_code

        user_data.covid_symptoms_score = user_data.covid_symptoms_score.interpolate('nearest')
        user_data.covid_symptoms_score = user_data.covid_symptoms_score.fillna(method='pad')

        for col in ['heart_rate', 'meanrr', 'mxdmn', 'sdnn', 'rmssd', 'pnn50',
                    'mode', 'amo', 'lf', 'hf', 'vlf', 'lfhf', 'total_power']:
            user_data[col] = user_data[col].interpolate()
            user_data[col] = user_data[col].fillna(method='bfill')

        # B-spline interpolate blood pressure data to smoothen
        for col in bp_cols:
            bp_data = user_data[col].dropna()
            if not bp_data.empty:
                t_intervals = bp_data.index.to_series().diff()
                if t_intervals.iloc[-1] > t_intervals.mean() + t_intervals.std():
                    bp_data = bp_data.iloc[:-1]

                k = 2
                if len(bp_data) <= k:
                    user_data[col] = user_data[col].interpolate()
                    user_data[col] = user_data[col].fillna(method='bfill')
                else:
                    epoch = pd.Timestamp('1970-01-01')
                    t_index = (bp_data.index - epoch) // pd.Timedelta('1s')

                    t, c, k = interp.splrep(t_index, bp_data.values, s=0, k=k)
                    xx = np.linspace(t_index.min(), t_index.max(), len(user_data[bp_data.index[0]:bp_data.index[-1]]))
                    spline = interp.BSpline(t, c, k, extrapolate=False)

                    bp_curve = spline(xx)
                    bp_curve = np.interp(bp_curve,
                                         (bp_curve.min(), bp_curve.max()),
                                         (bp_data.min(), bp_data.max()))

                    user_data[col] = np.nan
                    user_data.loc[bp_data.index[0]:bp_data.index[-1], col] = bp_curve

        # Outlier detection
        print("*Reutel Reutel* Doing Chauvenet outlier detection...")

        feature_cols = ['heart_rate', 'diastolic', 'systolic', 'meanrr', 'mxdmn', 'sdnn', 'rmssd',
                        'pnn50', 'mode', 'amo', 'lf', 'hf', 'vlf', 'lfhf', 'total_power']
        for col in feature_cols:
            print(f'Measurement is now: {col}')
            user_data = OutlierDistr.chauvenet(user_data, col)
            user_data.loc[user_data[f'{col}_outlier'], col] = np.nan
            del user_data[col + '_outlier']

        # Lowpass filtering (and even more interpolation!)
        for col in feature_cols:
            user_data = MisVal.impute_interpolate(user_data, col)

        fs = float(1000) / epsilon
        cutoff = 1.5
        for col in feature_cols:
            user_data = LowPass.low_pass_filter(user_data, col, fs, cutoff, order=10)
            user_data[col] = user_data[col + '_lowpass']
            del user_data[col + '_lowpass']

        for col in feature_cols:
            user_data = MisVal.impute_interpolate(user_data, col)

        # Add frequency features
        print("*Prrrt Prrrt* Adding frequency features...")
        if 'datetime' in user_data.columns:
            user_data = user_data.set_index('datetime', drop=True)
        user_data.index = pd.to_datetime(user_data.index)

        ws = int(float(0.5 * 60000) / epsilon)
        fs = float(1000) / epsilon

        for col in feature_cols:
            if not user_data[col].isnull().values.all():
                aggregations = user_data[col].rolling(f"{ws}s", min_periods=ws)
                user_data[col + '_temp_mean_ws_' + str(ws)] = aggregations.mean()
                user_data[col + '_temp_std_ws_' + str(ws)] = aggregations.std()

        user_data = CatAbs.abstract_categorical(user_data, ['covid_symptoms_score'], ['like'], 0.03,
                                                int(float(5 * 60000) / epsilon), 2)
        user_data = FreqAbs.abstract_frequency(copy.deepcopy(user_data), feature_cols,
                                               int(float(10000) / epsilon), float(1000) / epsilon)

        # Plot everything
        plot_covid_data(user_data, ['heart_rate', 'diastolic', 'systolic'],
                        ['heart rate', 'bp diastolic', 'bp systolic'], user_code,
                        f"./user_data/blood_pressure_{user_code}_interp.png")

        print(f"\n\nCorrelations for user {user_code} ({len(user_data)})")
        pearson_corr = user_data.corr('pearson')['covid_symptoms_score'][:]
        spearman_corr = user_data.corr('spearman')['covid_symptoms_score'][:]
        corrs = pd.concat([pearson_corr, spearman_corr], keys=['Pearson', 'Spearman'], axis=1)\
            .sort_values(['Pearson', 'Spearman'], ascending=False)

        print("Name,Pearson,Spearman")
        for corr in corrs.iterrows():
            print(f"{corr[0]},{corr[1].Pearson},{corr[1].Spearman}")

        user_data.to_csv(f"./user_data/covid_data_{user_code}.csv")


files = [os.path.join("./user_data/", file) for file in os.listdir("./user_data/")]
all_users_data = pd.concat((pd.read_csv(f) for f in files if f.endswith('csv')), ignore_index=True)
occ_most = int(all_users_data.covid_symptoms_score.mode().iloc[0])
all_counts = all_users_data['covid_symptoms_score'].value_counts()
print(all_counts)
all_users_data["major_baseline"] = occ_most
all_users_data["random_baseline"] = 0
all_users_data.to_csv(f"./all_users_data.csv")

clas_report = classification_report(all_users_data['covid_symptoms_score'], all_users_data['major_baseline'])
print(clas_report)        

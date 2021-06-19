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

granularity = 1
delta = 500
epsilon = delta / (1 + granularity)
for user_code, user_data in combi_data.groupby('user_code'):
    if len(user_data) > 99 and len(np.unique(user_data.covid_symptoms_score)) > 2:
        user_data = user_data.copy()
        dt_range_index = pd.date_range(user_data.index.min(), periods=len(user_data), freq=f"{delta}ms")
        user_data.index = dt_range_index

        plt.plot(user_data.index, user_data.heart_rate, label='heart rate')
        plt.plot(user_data.index, user_data.diastolic, label='bp diastolic')
        plt.plot(user_data.index, user_data.systolic, label='bp systolic')
        plt.plot(user_data.index, user_data.covid_symptoms_score, label='covid_symptoms_score')

        plt.title(user_code)
        plt.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2, fancybox=True, shadow=True)
        plt.show()

        # Remove duplicated blood pressure data
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

        # Interpolate blood pressure data to smoothen
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

                    user_data[bp_data.index[0]:bp_data.index[-1]][col] = bp_curve

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
        # for col in feature_cols:
        #     user_data = MisVal.impute_interpolate(user_data, col)
        #
        # fs = float(1000) / epsilon
        # cutoff = 1.5
        # for col in feature_cols:
        #     data = LowPass.low_pass_filter(user_data, col, fs, cutoff, order=10)
        #     data[col] = data[col + '_lowpass']
        #     del data[col + '_lowpass']
        #
        # for col in feature_cols:
        #     data = MisVal.impute_interpolate(user_data, col)
        #
        # n_pcs = np.argmax(PCA.determine_pc_explained_variance(user_data, feature_cols)) + 1
        # data = PCA.apply_pca(copy.deepcopy(user_data), feature_cols, n_pcs)

        # Add frequency features
        # print("*Prrrt Prrrt* Adding frequency features...")
        # if 'datetime' in data.columns:
        #     data = data.set_index('datetime', drop=True)
        # data.index = pd.to_datetime(data.index)
        #
        # ws = int(float(0.5 * 60000) / epsilon)
        # fs = float(1000) / epsilon
        #
        # for col in feature_cols:
        #     aggregations = user_data[col].rolling(f"{ws}s", min_periods=ws)
        #     user_data[col + '_temp_mean_ws_' + str(ws)] = aggregations.mean()
        #     user_data[col + '_temp_std_ws_' + str(ws)] = aggregations.std()
        #
        # user_data = CatAbs.abstract_categorical(user_data, ['covid_symptoms_score'], ['like'], 0.03,
        #                                         int(float(5 * 60000) / epsilon), 2)
        # user_data = FreqAbs.abstract_frequency(copy.deepcopy(user_data), feature_cols,
        #                                        int(float(10000) / epsilon), float(1000) / epsilon)

        # Clustering

        # Saving the world from covid using sick multi-user ML

        fig, axs = plt.subplots(2, figsize=(8, 5), sharex='all', gridspec_kw={'height_ratios': [5, 1]})

        axs[0].plot(user_data.index, user_data.heart_rate, label='heart rate')
        axs[0].plot(user_data.index, user_data.diastolic, label='bp diastolic interp.')
        axs[0].plot(user_data.index, user_data.systolic, label='bp systolic interp.')
        axs[0].set_title('Measurements', loc='left')
        axs[0].legend(loc='center left', bbox_to_anchor=(1.04, .5), ncol=1, fancybox=True, shadow=True, borderaxespad=0)

        axs[1].plot(user_data.index, user_data.covid_symptoms_score, label='covid_symptoms_score')
        axs[1].set_ylim((-1, 7))
        axs[1].set_title('Covid symptoms score', loc='left')

        fig.suptitle(f"User {user_code}")
        plt.show()

        print(f"\n\nCorrelations for user {user_code} ({len(user_data)})")
        pearson_corr = user_data.corr('pearson')['covid_symptoms_score'][:]
        spearman_corr = user_data.corr('spearman')['covid_symptoms_score'][:]
        corrs = pd.concat([pearson_corr, spearman_corr], keys=['Pearson', 'Spearman'], axis=1).sort_values(
            ['Pearson', 'Spearman'], ascending=False)

        print("Name,Pearson,Spearman")
        for corr in corrs.iterrows():
            print(f"{corr[0]},{corr[1].Pearson},{corr[1].Spearman}")

        user_data.to_csv(f"./user_data/covid_data_{user_code}.csv")

# CALCULATE BASELINE HERE
# 1. Haal alle CSV'tjes op die worden aangemaakt op bovenstaande line en plak ze achter elkaar
files = [os.path.join("./user_data/", file) for file in os.listdir("./user_data/")]
all_users_data = pd.concat((pd.read_csv(f) for f in files if f.endswith('csv')), ignore_index=True)
# 2. Haal totaal aantal covid_symtoms_scores op en noteer deze.
occ_most = int(all_users_data.covid_symptoms_score.mode().iloc[0])

# 3. Maak een random en majority baseline aan
all_users_data["major_baseline"] = occ_most
all_users_data["random_baseline"] = np.random.randint(1, 7, all_users_data.shape[0])
all_users_data.to_csv(f"./all_users_data.csv")

# 4. Bereken de precission-, recall- en F1-scores op basis van bovenstaande baseline
class_report_major = classification_report(all_users_data['covid_symptoms_score'], all_users_data['major_baseline'])
class_report_random = classification_report(all_users_data['covid_symptoms_score'], all_users_data['random_baseline'])
print('Classification Report Majority Baseline: \n', class_report_major)
print('Classification Report Random Baseline: \n', class_report_random)
      


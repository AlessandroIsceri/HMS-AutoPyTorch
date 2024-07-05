from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import warnings
import psutil
import multiprocessing
import sklearn.model_selection
import math


#warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def load_data(namespace):
    
    # LOAD TRAIN DATA
    data_train = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
    TARGETS = data_train.columns[-6:]
    print('Train shape:', data_train.shape )
    print('Targets', list(TARGETS))

    # aggregate data by eeg_id computing subset interval ([minOffset, maxOffset])
    train = data_train.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
        {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
    train.columns = ['spec_id','min']

    tmp = data_train.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
        {'spectrogram_label_offset_seconds':'max'})
    train['max'] = tmp

    # add patient_id to the train dataframe
    tmp = data_train.groupby('eeg_id')[['patient_id']].agg('first')
    train['patient_id'] = tmp

    # sum targets value for each eeg_id
    tmp = data_train.groupby('eeg_id')[TARGETS].agg('sum')
    for t in TARGETS:
        train[t] = tmp[t].values

    # calculate the relative frequency of every classification based on the total number of votes
    y_data = train[TARGETS].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train[TARGETS] = y_data

    # add most voted column to the train dataframe 
    tmp = data_train.groupby('eeg_id')[['expert_consensus']].agg('first')
    train['target'] = tmp

    # reset train index
    train = train.reset_index()
    print("Number of rows: ", len(train))
    print('Train non-overlapp eeg_id shape:', train.shape )
    print(train.head())

    # we are using two datasets created by Chris Deotte [https://www.kaggle.com/datasets/cdeotte/brain-spectrograms],
    # [https://www.kaggle.com/datasets/cdeotte/brain-eeg-spectrograms]
    # which contains the raw eeg waveform converted into a spectrogram
    spectrograms = np.load('/kaggle/input/brain-spectrograms/specs.npy',allow_pickle=True).item()
    all_eegs = np.load('/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy',allow_pickle=True).item()

    # FEATURE NAMES
    PATH = '/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/'
    SPEC_COLS = pd.read_parquet(f'{PATH}1000086677.parquet').columns[1:]
    FEATURES = [f'{c}_mean_10m' for c in SPEC_COLS]
    FEATURES += [f'{c}_min_10m' for c in SPEC_COLS]
    FEATURES += [f'{c}_mean_20s' for c in SPEC_COLS]
    FEATURES += [f'{c}_min_20s' for c in SPEC_COLS]
    FEATURES += [f'eeg_mean_f{x}_10s' for x in range(512)]
    FEATURES += [f'eeg_min_f{x}_10s' for x in range(512)]
    FEATURES += [f'eeg_max_f{x}_10s' for x in range(512)]
    FEATURES += [f'eeg_std_f{x}_10s' for x in range(512)]

    print(f'We are creating {len(FEATURES)} features for {len(train)} rows... ',end='')

    data = np.zeros((len(train),len(FEATURES)))
    for k in range(len(train)):
        if k%100==0: 
            print(k,', ',end='')
        row = train.iloc[k]
        r = int( (row['min'] + row['max'])//4 ) 

        # 10 MINUTE WINDOW FEATURES (MEANS and MINS)
        x = np.nanmean(spectrograms[row.spec_id][r:r+300,:],axis=0)
        data[k,:400] = x
        x = np.nanmin(spectrograms[row.spec_id][r:r+300,:],axis=0)
        data[k,400:800] = x

        # 20 SECOND WINDOW FEATURES (MEANS and MINS)
        x = np.nanmean(spectrograms[row.spec_id][r+145:r+155,:],axis=0)
        data[k,800:1200] = x
        x = np.nanmin(spectrograms[row.spec_id][r+145:r+155,:],axis=0)
        data[k,1200:1600] = x

        # RESHAPE EEG SPECTROGRAMS 128x256x4 => 512x256
        eeg_spec = np.zeros((512,256),dtype='float32')
        xx = all_eegs[row.eeg_id]
        for j in range(4): 
            eeg_spec[128*j:128*(j+1),] = xx[:,:,j]

        # 10 SECOND WINDOW FROM EEG SPECTROGRAMS 
        x = np.nanmean(eeg_spec.T[100:-100,:],axis=0)
        data[k,1600:2112] = x
        x = np.nanmin(eeg_spec.T[100:-100,:],axis=0)
        data[k,2112:2624] = x
        x = np.nanmax(eeg_spec.T[100:-100,:],axis=0)
        data[k,2624:3136] = x
        x = np.nanstd(eeg_spec.T[100:-100,:],axis=0)
        data[k,3136:3648] = x

    train[FEATURES] = data
    print(); print('New train shape:',train.shape)

    # remove rows with NaN values
    nan_rows = train.isna().any(axis = 1)
    for i in range(len(nan_rows)):
        if nan_rows[i] == True:
            print(nan_rows[i], ' - ',i)
            for j in range(len(train.columns)):
                if train.iloc[:, j].name == 'target':
                    continue
                if math.isnan(train.iat[i,j]):
                    train.iat[i,j] = 0.0
    
    namespace.X = train[FEATURES]
    namespace.y = train[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
    print("X.shape: ", namespace.X.shape)
    print("y.shape: ", namespace.y.shape)
    
    print('sta per terminare il sottoproc, RAM memory % used:', psutil.virtual_memory()[2])


#create sub-process
manager = multiprocessing.Manager()

namespace = manager.Namespace()
namespace.X = X = pd.DataFrame() 
namespace.y = y = pd.DataFrame()

p = multiprocessing.Process(target=load_data, args=(namespace,))
p.start()
p.join()
print('dopo aver terminato il sottoproc,RAM memory % used:', psutil.virtual_memory()[2])

print("dividendo tra X_train...")

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    namespace.X,
    namespace.y,
    random_state=1,
    train_size = 0.90,
    test_size = 0.10
)

targets = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    
def runModel(y_train, X_train, y_test, X_test):

    # Initialize CatBoostRegressor
    model = CatBoostRegressor(iterations=499,
                              learning_rate=0.0060,
                              depth=11,
                              task_type="GPU",
                              devices='0:1',
                              loss_function= 'MultiRMSE', 
                              eval_metric= 'MultiRMSE', )
    # Fit model
    model.fit(X_train, y_train)
    # Get predictions
    y_pred = model.predict(X_test)
    
    y_pred_df = pd.DataFrame(y_pred.copy(), columns = targets)
    y_pred_df['id'] = np.arange(len(y_pred))

    y_test_df = pd.DataFrame(y_test.copy())
    y_test_df['id'] = np.arange(len(y_test))
    
    print("y_test_df: ", y_test_df)
    print("y_pred_df: ", y_pred_df)

    print("CatBoost Starter - By Alessandro Isceri & Mattia Ingrassia")
    
    for i in range(6):
        print("R^2 ", targets[i], " : ", sklearn.metrics.r2_score(y_test_df[[targets[i]]], y_pred_df[[targets[i]]]))    
    
runModel(y_train, X_train, y_test, X_test) 

# HMS - AutoPyTorch

Il progetto consiste nell’analizzare un dataset contenente segnali elettroencefalografici (EEG) e spettrogrammi utilizzando AutoML come tecnica di Machine Learning.

Nello specifico, il dataset utilizzato è stato preso da una competizione di [Kaggle](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview) e il framework AutoML usato è [AutoPyTorch](https://github.com/automl/Auto-PyTorch).

Il dataset contiene le valutazioni fornite da alcuni esperti sulla base di diversi campioni EEG della durata di 50 secondi con i rispettivi spettrogrammi ricavati da pazienti ospedalieri critici.

L’obiettivo è stimare, attraverso l’uso della regressione, sei valori target:

* seizure_vote
* lpd_vote&nbsp;&rarr;&nbsp;lpd = lateralized periodic discharges
* gpd_vote&nbsp;&rarr;&nbsp;gpd = generalized periodic discharges
* lrda_vote&nbsp;&rarr;&nbsp;lrda = lateralized rhythmic delta activity
* grda_vote&nbsp;&rarr;&nbsp;grda = generalized rhythmic delta activity 
* other_vote			

Ognuno di questi numeri reali rappresenta la frequenza relativa dei voti ottenuti per ciascuna categoria, ed è dunque un numero compreso tra 0 ed 1.

Inoltre, la somma delle frequenze relative deve essere pari ad 1.

## Getting started

E' possibile visualizzare i notebook su Kaggle con i link nelle rispettive sezioni.

Tuttavia, se si desidera eseguire il codice sulla propria macchina, è necessario:
* Installare le librerie necessarie per l'esecuzione del codice
* Scaricare i dataset utilizzati dai notebook su Kaggle (si possono scaricare dai notebook stessi)

### CatBoost

Il link per il notebook di Kaggle che utilizza CatBoost è disponibile [qui](https://www.kaggle.com/code/alessandroisceri/catboost-hms).

Nel caso non fosse già installata, si può scaricare la libreria tramite il seguente comando

```shell
  pip install catboost
```

Se non si dispone di una GPU, è possibile modificare la seguente porzione di codice

https://github.com/Alessandro-Isceri/HMS-AutoPyTorch/blob/09c39f8b5f3970eebbfdfc6c7c0a889a8c006f71/src/HMSCatBoost.py#L161-L168

Sostituendo 
```python
    task_type="GPU",
    devices='0:1'
```
con
```python
    task_type="CPU"
```

### AutoPyTorch

Il link per il notebook di Kaggle che utilizza AutoPyTorch è disponibile [qui](https://www.kaggle.com/code/alessandroisceri/autopytorch-hms).

Nel momento in cui è stato sviluppato questo progetto, la normale installazione di AutoPyTorch creava dei problemi con le versioni di python successive alla 3.9.

Per riuscire ad installare e ad utilizzare correttamente AutoPyTorch si può procedere in questi due modi:

#### 1) Utilizzo di un virtual environment

Con python 3.9 i problemi creati dalle dipendenze di AutoPyTorch non sussistono, dunque creando un virtual environment si può facilmente risolvere il problema.
```shell
apt-get update
apt install -y python3.9
pip install virtualenv
virtualenv venv -p $(which python3.9)
/venv/bin/pip install autopytorch
```
Nel codice utilizzato in questo progetto è stato necessario anche installare le seguenti librerie per interagire con i file che contenevano i dati
```shell
/venv/bin/pip install pyarrow
/venv/bin/pip install fastparquet
```
Una volta che il setup dell'ambiente di lavoro è terminato, è possibile proseguire salvando il proprio codice in un file python (in questo caso HMSAutoPyTorch.py) ed eseguirlo con il seguente comando
```shell
/venv/bin/python3.9 HMSAutoPyTorch.py
```
#### 2) Installazione di AutoPyTorch tramite git

Grazie all'intervento di un altro utente ([Borda](https://github.com/Borda)), che ha modificato le dipendenze come è possibile vedere [qui](https://github.com/automl/Auto-PyTorch/pull/506), è possibile utilizzare il seguente comando per installare AutoPyTorch
```shell
pip install "git+https://github.com/Borda/Auto-PyTorch.git@bump/sklearn-1.0+"
```
### Extra

Una documentazione più approfondita del lavoro è visualizzabile nel file [D10 - AutoML for Brain Predictor.pdf](https://github.com/Alessandro-Isceri/HMS-AutoPyTorch/blob/main/D10%20-%20AutoML%20for%20Brain%20Predictor.pdf)


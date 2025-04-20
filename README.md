# Analisi dell’andamento della temperatura globale dal 1880 a oggi
Questo progetto analizza l'andamento delle temperature globali utilizzando dati ufficiali NASA (GISTEMP).  
Viene mostrata una regressione lineare sia sull'intero periodo (1880-oggi) che sugli ultimi 30 anni.

## Obiettivo
Stimare l'innalzamento medio annuo della temperatura globale e visualizzare i cambiamenti nel tempo.

## Dataset
- Fonte: [NASA GISTEMP](https://data.giss.nasa.gov/gistemp/)
- File: `GLB.Ts+dSST.csv`

## Nota
Il programma lavora con anomalie di temperatura, cioè la differenza rispetto a una media di riferimento (solitamente il periodo 1951–1980).
Valori negativi indicano anni più freddi della media, valori positivi anni più caldi.

## Librerie usate
- pandas per la manipolazione dei dati
- matplotlib.pyplot per i grafici
- sklearn.linear_model.LinearRegression per la regressione lineare

#!/bin/bash

OUTPUT_CSV="metricas_sigmoid.csv"
RUNS=30

# Cabeçalho
echo "Run,Accuracy,Precision,Recall,F1,Treinamento" > $OUTPUT_CSV

# Somatórios
sum_acc=0
sum_prec=0
sum_rec=0
sum_f1=0
sum_tt=0

for ((i=1; i<=RUNS; i++))
do
    echo "Rodada $i"

    # Executa e captura só a linha com as métricas
    OUTPUT=$(./weather_forecast | head -n 1)

    # Separa por vírgulas
    IFS=',' read -r ACC PREC REC F1 TT <<< "$OUTPUT"

    echo "$i,$ACC,$PREC,$REC,$F1,$TT" >> $OUTPUT_CSV

    # Acumuladores
    sum_acc=$(echo "$sum_acc + $ACC" | bc)
    sum_prec=$(echo "$sum_prec + $PREC" | bc)
    sum_rec=$(echo "$sum_rec + $REC" | bc)
    sum_f1=$(echo "$sum_f1 + $F1" | bc)
    sum_tt=$(echo "$sum_tt + $TT" | bc)
done

# Médias
avg_acc=$(echo "scale=6; $sum_acc / $RUNS" | bc)
avg_prec=$(echo "scale=6; $sum_prec / $RUNS" | bc)
avg_rec=$(echo "scale=6; $sum_rec / $RUNS" | bc)
avg_f1=$(echo "scale=6; $sum_f1 / $RUNS" | bc)
avg_tt=$(echo "scale=6; $sum_tt / $RUNS" | bc)

echo "Average,$avg_acc,$avg_prec,$avg_rec,$avg_f1,$avg_tt" >> $OUTPUT_CSV
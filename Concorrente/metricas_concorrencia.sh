#!/bin/bash

OUTPUT_CSV="metricas_concorrencia_teste.csv"
RUNS=30

# Cabeçalho
echo "Run,TS,TC,TT" > $OUTPUT_CSV

# Somatórios
sum_ts=0
sum_tc=0
sum_tt=0

for ((i=1; i<=RUNS; i++))
do
    echo "Rodada $i"

    # Executa e captura as duas primeiras linhas de saída (TS e TC)
    OUTPUT=$(./weather_forecast | grep -E "TS:|TC:")

    TS=$(echo "$OUTPUT" | grep "TS:" | cut -d':' -f2 | xargs)
    TC=$(echo "$OUTPUT" | grep "TC:" | cut -d':' -f2 | xargs)

    # Executa o normal e pega só a linha com as métricas
    TRAINING=$(./weather_forecast | head -n 1 | cut -d',' -f5)

    echo "$i,$TS,$TC,$TRAINING" >> $OUTPUT_CSV

    # Acumuladores (usando bc com scale para precisão)
    sum_ts=$(echo "$sum_ts + $TS" | bc)
    sum_tc=$(echo "$sum_tc + $TC" | bc)
    sum_tt=$(echo "$sum_tt + $TRAINING" | bc)
done

# Médias
avg_ts=$(echo "scale=8; $sum_ts / $RUNS" | bc)
avg_tc=$(echo "scale=8; $sum_tc / $RUNS" | bc)
avg_tt=$(echo "scale=8; $sum_tt / $RUNS" | bc)

echo "Average,$avg_ts,$avg_tc,$avg_tt" >> $OUTPUT_CSV
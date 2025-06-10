#!/bin/bash

OUTPUT_CSV="metrics_Leaky_ReLU.csv"
RUNS=30

# Cabeçalho do CSV
echo "Run,Accuracy,Precision,Recall,F1,Tempo de Treinamento" > $OUTPUT_CSV

# Variáveis para somar métricas e calcular médias
sum_acc=0
sum_prec=0
sum_rec=0
sum_f1=0
sum_tt=0

for ((i=1; i<=RUNS; i++))
do
    echo "Rodada $i"
    
    # Executa seu programa e captura a saída
    OUTPUT=$(./weather_forecast)
    
    # Extrai as métricas da saída (assumindo formato fixo)
    ACC=$(echo "$OUTPUT" | grep -i "Acurácia" | awk '{print $2}')
    PREC=$(echo "$OUTPUT" | grep -i "Precisão" | awk '{print $2}')
    REC=$(echo "$OUTPUT" | grep -i "Recall" | awk '{print $2}')
    F1=$(echo "$OUTPUT" | grep -i "F1 Score" | awk '{print $3}')
    TT=$(echo "$OUTPUT" | grep -i "Tempo de Treinamento" | awk '{print $4}')
    
    # Adiciona ao CSV
    echo "$i,$ACC,$PREC,$REC,$F1,$TT" >> $OUTPUT_CSV
    
    # Acumula para média
    sum_acc=$(echo "$sum_acc + $ACC" | bc)
    sum_prec=$(echo "$sum_prec + $PREC" | bc)
    sum_rec=$(echo "$sum_rec + $REC" | bc)
    sum_f1=$(echo "$sum_f1 + $F1" | bc)
    sum_tt=$(echo "$sum_tt + $TT" | bc)
done

# Calcula médias
avg_acc=$(echo "scale=6; $sum_acc / $RUNS" | bc)
avg_prec=$(echo "scale=6; $sum_prec / $RUNS" | bc)
avg_rec=$(echo "scale=6; $sum_rec / $RUNS" | bc)
avg_f1=$(echo "scale=6; $sum_f1 / $RUNS" | bc)
avg_tt=$(echo "scale=6; $sum_tt / $RUNS" | bc)

# Adiciona média ao CSV
echo "Average,$avg_acc,$avg_prec,$avg_rec,$avg_f1,$avg_tt" >> $OUTPUT_CSV

echo "Execuções finalizadas. Resultados salvos em $OUTPUT_CSV"
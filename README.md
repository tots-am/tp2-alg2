# TP2 – Algoritmos 2 (DCC207): TSP – B&B, Twice-around-the-tree e Christofides

## Reproduzir
make venv          # ambiente Python
make heuristics    # heurísticas, features, tours de warm-start (obrigatório antes do bnb)
make scaling       # curvas de complexidade (N até 1000)
make bnb           # B&B, 150 execuções (~5 h, limite de 30 min cada)
make analysis      # figuras e tabelas em results/analysis/
make report        # compila o relatório

## Estrutura
src/bnb/bnb.cpp (núcleo C++11), src/tsp/ (heurísticas, features, instâncias),
src/run_experiments.py, src/analysis.py, results/ (dados), report/ (artigo e anexo).

## Observação
A coluna maxrss_kb de results/bnb.csv está incorreta (herdada do processo Python);
ver Seção 2 do relatório. Status "NA" = tempo esgotado (30 min).
PY := .venv/bin/python
CXXFLAGS := -std=c++11 -O3 -march=native -Wall
WORKERS ?= 3
NICE := nice -n 10
TL ?= 1800

src/bnb/bnb: src/bnb/bnb.cpp
	g++ $(CXXFLAGS) -o $@ $<

venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

heuristics: src/bnb/bnb
	$(NICE) $(PY) -m src.run_experiments heuristics

# rodada principal: cotas hk e mst em todas as instâncias, 5 warm-starts
bnb: src/bnb/bnb
	$(NICE) $(PY) -m src.run_experiments bnb --bounds hk,mst --time-limit $(TL) --workers $(WORKERS)

# ablação da cota ingênua (só instâncias pequenas)
ablation: src/bnb/bnb
	$(NICE) $(PY) -m src.run_experiments bnb --bounds naive --max-n 30 --time-limit 300 --workers $(WORKERS)

# acompanhamento da rodada do B&B (só leitura, não interfere na execução)
status:
	@echo "concluídos: `tail -n +2 results/bnb.csv | wc -l` de `grep -o '^[0-9]* jobs' results/bnb_run.log | tail -1`"
	@echo "rodando agora:"
	@ps -eo etime,args | grep '[b]nb/bnb --' | awk '{print "  " $$1 " decorridos, cota " $$4}'
	@if pgrep -f run_experiments >/dev/null; then echo "situação: EM EXECUÇÃO"; else echo "situação: TERMINOU (ou foi interrompida)"; fi
	@echo "últimos resultados:"
	@tail -n 5 results/bnb_run.log

.PHONY: venv heuristics bnb ablation status

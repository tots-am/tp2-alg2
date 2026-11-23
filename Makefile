PY := .venv/bin/python
CXXFLAGS := -std=c++11 -O3 -march=native -Wall
WORKERS ?= 3
NICE := nice -n 10
TL_HK ?= 1800
TL_MST ?= 1800

src/bnb/bnb: src/bnb/bnb.cpp
	g++ $(CXXFLAGS) -o $@ $<

venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

heuristics: src/bnb/bnb
	$(NICE) $(PY) -m src.run_experiments heuristics

# rodada principal: cotas hk e mst em todas as instâncias, 5 warm-starts
bnb: src/bnb/bnb
	$(NICE) $(PY) -m src.run_experiments bnb --bounds hk --time-limit $(TL_HK) --workers $(WORKERS)
	$(NICE) $(PY) -m src.run_experiments bnb --bounds mst --time-limit $(TL_MST) --workers $(WORKERS)

# ablação da cota ingênua (só instâncias pequenas)
ablation: src/bnb/bnb
	$(NICE) $(PY) -m src.run_experiments bnb --bounds naive --max-n 30 --time-limit 300 --workers $(WORKERS)

.PHONY: venv heuristics bnb ablation

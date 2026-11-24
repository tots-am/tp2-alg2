# Relatório (modelo SBC)

`sbc-template.sty` e `sbc.bst` são o modelo oficial da SBC (Hübner & Bordini, 2005),
copiados de https://github.com/uefs/sbc-template-latex.

## Compilar

```
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Pacotes do Ubuntu necessários: `texlive-latex-extra texlive-lang-portuguese`.
As figuras e tabelas são lidas diretamente de `../results/analysis/` (macro `\resdir`).

## Fluxo de atualização

- Figuras e tabelas: `.venv/bin/python -m src.analysis` (na raiz; regenera `results/analysis/`).
- Trechos marcados com `\todo{...}` (em vermelho no PDF) são os pontos a revisar e completar.

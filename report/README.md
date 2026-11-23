# Relatório (modelo SBC)

Não há LaTeX instalado na máquina; a forma recomendada de compilar é o **Overleaf**.

1. No Overleaf, crie um projeto a partir do modelo **"SBC Conferences Template"**
   (galeria SBC: https://www.overleaf.com/gallery/tagged/sbc). Ele traz `sbc-template.sty` e `sbc.bst`.
2. Apague o `.tex`/`.bib` de exemplo e envie: `main.tex`, `refs.bib`, `anexo_prompts.tex`,
   `anexo_prompts_gerado.tex`.
3. Crie a pasta `figs/` no Overleaf e envie os PDFs de `results/analysis/figs/`.
   Para as tabelas, crie `results/analysis/tables/` no Overleaf, ou mude `\resdir` em `main.tex`
   para o caminho onde as enviou.
4. Compile `main.tex` (pdfLaTeX + BibTeX).

Alternativa local: `sudo apt install texlive-latex-extra texlive-lang-portuguese texlive-publishers`
e copiar `sbc-template.sty` e `sbc.bst` do projeto do Overleaf para esta pasta.

## Fluxo de atualização

- Figuras e tabelas: `.venv/bin/python -m src.analysis` (regenera `results/analysis/`).
- Anexo de prompts: `python3 tools/export_prompts.py` (regenera `anexo_prompts_gerado.tex`;
  edições manuais vão em `anexo_prompts.tex`).
- Trechos marcados com `\todo{...}` (em vermelho no PDF) são os pontos a revisar e completar.

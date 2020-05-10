SRC :=$(wildcard *.tex *.bib)

ProgressReport.pdf: $(SRC)
	pdflatex ProgressReport
	bibtex ProgressReport
	pdflatex ProgressReport

clean:
	rm -f *.glo *.log *.dvi *.gls *.toc *.aux *.ist *.out *.glg *.pdf *.bbl *.blg *.lof *.brf

.PHONY: clean

SRC :=$(wildcard *.tex *.bib)

FinalReport.pdf: $(SRC)
	pdflatex FinalReport
	bibtex FinalReport
	pdflatex FinalReport

clean:
	rm -f *.glo *.log *.dvi *.gls *.toc *.aux *.ist *.out *.glg *.pdf *.bbl *.blg *.lof *.brf

.PHONY: clean

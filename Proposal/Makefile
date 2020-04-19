SRC :=$(wildcard *.tex *.bib)

Proposal.pdf: $(SRC)
	pdflatex Proposal 
	bibtex Proposal
	pdflatex Proposal 

clean:
	rm -f *.glo *.log *.dvi *.gls *.toc *.aux *.ist *.out *.glg *.pdf *.bbl *.blg *.lof *.brf

.PHONY: clean

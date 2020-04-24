SRC := $(wildcard Slides.tex *.tex *.bib)

MLInterp.pdf: $(SRC)
	pdflatex Slides

clean:
	rm -rf *.glo *.log *.dvi *.gls *.toc *.aux *.ist *.out *.pdf *.bbl *.blg *.lof *.brf *.nav *.snm

.PHONY: clean

FNAME_BASE=yantosca_COSC6336_assg_1

all: doc

doc: doc/$(FNAME_BASE).pdf

%.pdf: %.tex %.bib
	@lualatex -shell-escape --output-directory=doc $*.tex

superclean: clean superclean-doc-$(FNAME_BASE)

superclean-doc-%:
	@rm -f doc/$*.pdf

clean: clean-doc-$(FNAME_BASE)
	@rm -f *~
	@rm -f src/*~

clean-doc-%: 
	@rm -f doc/$*.aux
	@rm -f doc/$*.bbl
	@rm -f doc/$*.bcf
	@rm -f doc/$*.log
	@rm -f doc/$*.run.xml
	@rm -f doc/$*.dvi
	@rm -f doc/$*.blg
	@rm -f doc/$*.auxlock
	@rm -f doc/$*.pyg
	@rm -f doc/$*-figure*
	@rm -f doc/$*.toc
	@rm -f doc/$*.out
	@rm -f doc/$*.snm
	@rm -f doc/$*.nav
	@rm -rf doc/_minted-$*
	@rm -f doc/*~

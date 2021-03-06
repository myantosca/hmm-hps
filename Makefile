FNAME_BASE=yantosca_COSC6336_assg_1

all: doc srcdist

srcdist:
	@rm -rf ghostant
	@mkdir -p ghostant
	@cp src/hmmpos.py ghostant
	@cp src/README.txt ghostant
	@cp -r dataset ghostant
	@tar cvzf yantosca_COSC6336_assg_1.tar.gz ghostant

doc: .EMPTY
	@make -C doc

superclean: clean superclean-doc-$(FNAME_BASE)

superclean-doc-%:
	@rm -f doc/$*.pdf

clean: clean-doc-$(FNAME_BASE)
	@rm -f *~
	@rm -f src/*~
	@rm -rf ghostant

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

.EMPTY:

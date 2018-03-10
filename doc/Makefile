FNAME_BASE=yantosca_COSC6336_assg_1

all: $(FNAME_BASE).pdf

%.pdf: %.tex %.bib
	@lualatex -shell-escape $*.tex
	@bibtex $*
	@lualatex -shell-escape $*.tex
	@lualatex -shell-escape $*.tex

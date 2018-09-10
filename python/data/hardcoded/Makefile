
.PHONY: pred run

all: clean run pred
pred: pred/status.csv
run: save/root.ini

pred/status.csv:	save/root.ini
	mkdir -p pred
	rm -f pred/*
	smurff --save-prefix pred --root save/root.ini --row-features feat_0_0.mtx

save/root.ini:
	mkdir -p save
	rm -f save/*
	smurff --ini macau.ini
	
clean:
	mkdir -p pred save
	rm -f save/* pred/*


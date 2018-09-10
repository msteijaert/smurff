
.PHONY: pred run

all: clean run pred
pred: pred/status.csv
run: save/root.ini

pred/status.csv:	save/root.ini
	rm -rf pred
	mkdir -p pred
	smurff --save-freq -1 --save-prefix pred --root save/root.ini --row-features feat_0_0.mtx --save-extension .csv

save/root.ini:
	rm -rf save
	mkdir -p save
	smurff --ini macau.ini
	
clean:
	rm -fr save pred
	mkdir -p pred save


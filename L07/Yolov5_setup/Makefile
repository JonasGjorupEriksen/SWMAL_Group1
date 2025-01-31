PYFILE=yolov5_demo.py
FIG=demo

run: clean
	$(PYFILE) Figs/$(FIG)
	xview Figs/$(FIG) &
	xview runs/detect/exp/$(FIG).jpg
	
edit:
	joe $(PYFILE)
	
clean:
	@ rm -rf runs/* # carefull with that ax, Eugene!

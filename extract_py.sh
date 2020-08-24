#!/bin/sh

for file in `find notebooks -maxdepth 1 -iname *.ipynb`; do
  echo $file;
  jupytext --to py --output "py/$(basename ${file%.*}).py" ${file}
done

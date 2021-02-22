#!/bin/bash
path=$(pwd)
echo -e "\n# Add astro_tools to PYTHONPATH" >> ~/.bashrc;
echo "export PYTHONPATH=$path:"'$PYTHONPATH' >> ~/.bashrc;
source ~/.bashrc;

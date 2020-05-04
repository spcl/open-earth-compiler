import subprocess
import os

files = os.listdir("./")

for file in files:
  if file == "convert.py":
    continue
  command = r"sed 's/temp/view/g' " + file
  command = command + " | ../../../build_master/bin/oec-opt --mlir-print-op-generic"
  command = command + " | sed  's/view/temp/g' "
  command = command + " | ../../build/bin/oec-opt > " + "conv_" + file
  print(command)
  subprocess.run(command, stderr=subprocess.STDOUT, shell=True)
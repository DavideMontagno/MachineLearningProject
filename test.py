import sys
import os
sys.path.append(os.path.abspath("../Model_2_PytorchNN"))
import model_2
sys.path.append(os.path.abspath("../Model_1_KerasNN"))
import model_1

print('Starting model 1')
model_1.cross_validation1()
print('Starting model 2')
model_2.cross_validation2()


from Lab2.Skin_NonSkin import Skin_NonSkin_experiment
from Lab2.blood import blood_exp
from Lab2.data_banknote_authentication import data_banknote_authentication_exp
from Lab2.design_experiment import design_experiment
from Lab2.heart import heart_exp

cycle_times = 1000000
descending_step_size = 0.1
iteration_error = 1e-5
design_experiment(200, 0, True, cycle_times, descending_step_size, iteration_error)
design_experiment(200, 0.001, True, cycle_times, descending_step_size, iteration_error)
design_experiment(200, 0, False, cycle_times, descending_step_size, iteration_error)
design_experiment(200, 0.001, False, cycle_times, descending_step_size, iteration_error)

cycle_times1 = 1000000
descending_step_size1 = 0.001
iteration_error1 = 1e-5
Skin_NonSkin_experiment(0, cycle_times1, descending_step_size1, iteration_error1)
Skin_NonSkin_experiment(0.01, cycle_times1, descending_step_size1, iteration_error1)

cycle_times3 = 1000000
descending_step_size3 = 0.1
iteration_error3 = 1e-5
blood_exp(0, cycle_times3, descending_step_size3, iteration_error3)
blood_exp(0.01, cycle_times3, descending_step_size3, iteration_error3)


cycle_times4 = 1000000
descending_step_size4 = 0.1
iteration_error4 = 1e-5
heart_exp(0, cycle_times4, descending_step_size4, iteration_error4)
heart_exp(0.01, cycle_times4, descending_step_size4, iteration_error4)

cycle_times2 = 1000000
descending_step_size2 = 0.1
iteration_error2 = 1e-5
data_banknote_authentication_exp(0, cycle_times2, descending_step_size2, iteration_error2)
data_banknote_authentication_exp(0.01, cycle_times2, descending_step_size2, iteration_error2)




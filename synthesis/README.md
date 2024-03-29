# Handwriting generation
CopyMonkey (this project) web app is hosted at: https://copymonkey.xyz
### Data description:

There are 2 data files that you need to consider: `data.npy` and `sentences.txt`. `data.npy`contains 6000 sequences of points that correspond to handwritten sentences. `sentences.txt` contains the corresponding text sentences. You can see an example on how to load and plot an example sentence in `example.ipynb`. Each handwritten sentence is represented as a 2D array with T rows and 3 columns. T is the number of timesteps. The first column represents whether to interrupt the current stroke (i.e. when the pen is lifted off the paper). The second and third columns represent the relative coordinates of the new point with respect to the last point. Please have a look at the plot_stroke if you want to understand how to plot this sequence.

### Unconditional generation.
Generated samples:
![alt text](https://github.com/swechhachoudhary/Handwriting-synthesis/blob/master/results/gen_prediction_samples/pred_10_400.png)
![alt text](https://github.com/swechhachoudhary/Handwriting-synthesis/blob/master/results/gen_prediction_samples/pred_4_700.png)
Samples generated using priming:
* Prime style text is "medical assistance", text after this is generated by model
![alt text](https://github.com/swechhachoudhary/Handwriting-synthesis/blob/master/results/gen_prediction_samples/pred_8_400_medical_assistance.png)
* Prime style text is "something which he is passing on", text after this is generated by model
![alt text](https://github.com/swechhachoudhary/Handwriting-synthesis/blob/master/results/gen_prediction_samples/pred_8_700_something_which_he_is_passing_on.png)
* Prime style text is "In Africa Jones hotels spring", text after this is generated by model
![alt text](https://github.com/swechhachoudhary/Handwriting-synthesis/blob/master/results/gen_prediction_samples/prime_1078_10_400_In%20Africa_Jones_hotels_spring.png)
### Conditional generation.
Generated samples:
![alt text](https://github.com/swechhachoudhary/Handwriting-synthesis/blob/master/results/gen_synthesis_samples/syn_10_python_handwriting_synthesis_224.png)
![alt text](https://github.com/swechhachoudhary/Handwriting-synthesis/blob/master/results/gen_synthesis_samples/syn_10_urnn.png)
![alt text](https://github.com/swechhachoudhary/Handwriting-synthesis/blob/master/results/gen_synthesis_samples/syn_10_by_Swechha.png)

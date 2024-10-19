<h1 style="background-color:#d2d1e3;border-radius:8px">Physical Reasoning Challenge: Readme</h1>

<h3 style="color:#363636">Where to find the training and test data.</h3>
<h5 style="background-color:#fff0f6;border-radius:8px"><b>Training dataset (<span style="color:grey">COMP90086_2024_Project_train</span>):</b> <br/>This folder has to be accessible in the current folder (<span style="color:grey">final_submission</span>).</h5>

<h5 style="background-color:#fff5ec;border-radius:8px"><b>Test dataset (<span style="color:grey">COMP90086_2024_Project_test</span>):</b> <br/>This folder has to be accessible in the current folder (<span style="color:grey">final_submission</span>).</h5>

<h3 style="color:#363636">Prediction file</h3>
<h5>The predictions are stored into <span style="color:grey">predicted_stable_heights.csv</span>. This file must be located at the root of the <span style="color:grey">final_submission</span> folder.</h5>

<h3 style="color:#363636">Inception_V3.ipynb</h3>
<h5>This file contains the code used to build and train the model. When run, this file modifies the prediction file (<span style="color:grey">predicted_stable_heights.csv</span>).</h5>

<h3 style="color:#363636">Validation_accuracy_analysis.ipynb</h3>
<h5>This file is a variant of the previous one (<span style="color:grey">Inception_V3.ipynb</span>) except that it doesn't modify the prediction file (<span style="color:grey">predicted_stable_heights.csv</span>) and computes the validation accuracy of the model on an unseen part onn the training set.</h5>
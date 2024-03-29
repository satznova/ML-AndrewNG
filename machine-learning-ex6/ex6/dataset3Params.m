function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

param_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
param_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

% Initialising error to Infinity
error = Inf

for c = param_C
    for sig = param_sigma
        
        % Train the model with dataset X and y
        model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));

        % Make predictions with cross-validation dataset for the built SVM model
        predictions = svmPredict(model, Xval);

        c_sig_error = mean(double(predictions ~= yval));  

        if c_sig_error < error
            error = c_sig_error;
            C = c;
            sigma = sig;
        endif
    endfor
endfor


% =========================================================================

end

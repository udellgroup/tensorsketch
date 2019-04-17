%% Demo 2: Tucker decomposition of dense tensor
%
% This script gives a demo of tucker_ts and tucker_ttmts decomposing a 
% dense tensor. The script generates a dense tensor and then decomposes 
% it using both tucker_ts and tucker_ttmts, as well as tucker_als from 
% Tensor Toolbox [1]. Please note that the script requires Tensor Toolbox
% version 2.6 or later.
%
% REFERENCES:
%
%   [1] B. W. Bader, T. G. Kolda and others. MATLAB Tensor Toolbox 
%       Version 2.6, Available online, February 2015. 
%       URL: http://www.sandia.gov/~tgkolda/TensorToolbox/.

% Author:   Osman Asif Malik
% Email:    osman.malik@colorado.edu
% Date:     September 17, 2018

%% Include relevant files

addpath(genpath('help_functions'));

%% Setup
I = [100 100 100]; 
noise_level = 0.1; % Amount of noise added to nonzero elements
tol = 1e-4; % Tolerance
maxiters = 100; % Maximum number of iterations

R_true = [5 5 5] ; % True tensor rank

%% Generate random dense tensor
G_true = tensor(rand(R_true)); 
A_true = cell(length(R_true),1);
for k = 1:length(R_true)
    A_true{k} = randn(I(k),R_true(k));
    [Qfac, ~] = qr(A_true{k}, 0);
    A_true{k} = Qfac;
end
%Y = tensor(ttensor(G_true, A_true)) + noise_level*randn(I); 
% Change to the noise level to the problem I use.
Y = tensor(ttensor(G_true, A_true));  
true_signal_mag = norm(Y)^2; 
Y = Y + noise_level*randn(I)*sqrt(noise_level^2*true_signal_mag/100^3);

R = [5 5 5]; 
Y_tucker_als = tucker_als(Y, R, 'tol', tol, 'maxiters', maxiters);
norm(Y- tensor(Y_tucker_als))
norm(tensor(Y_tucker_als))
normY
tucker_als_error = norm(Y - tensor(Y_tucker_als))/normY




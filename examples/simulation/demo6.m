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
tol = 1e-3; % Tolerance
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
true_signal_mag = norm(Y)^2
sqrt(true_signal_mag)
norm(reshape(noise_level*randn(I)*sqrt(noise_level^2*true_signal_mag/100^3), prod(I),1))
Y = Y + noise_level*randn(I)*sqrt(noise_level^2*true_signal_mag/100^3);
norm(reshape(Y, [prod(I) 1]))

Y = zeros(100,100,100);
for i = 1:5
    Y(i,i,i) = 1; 
end
for i = 6:100
    Y(i,i,i) = 1/(i-4); 
end


Rs = 5:2:15; 
err_mat = zeros(3,length(Rs));
for i = 1:length(Rs)
    %R_true =  [10 10 10]; % True tensor rank
    %R = [10 10 10]; % Algorithm target rank
    r = Rs(i);
    R = ones(1,3)*r; % Algorithm target rank
    %I = [400 400 400]; % Tensor size
    
    K = 2*r; % Sketch dimension parameter
    J1 = K; % First sketch dimension
    J2 = 2*K+1; % Second sketch dimension
   
    %% Algorithm
    [G_ts, A_ts] = tucker_ts(Y, R, J1, J2, 'tol', tol, 'maxiters', maxiters, 'verbose', false);
    [G_ttmts, A_ttmts] = tucker_ttmts(Y, R, J1, J2, 'tol', tol, 'maxiters', maxiters, 'verbose', false);
    Y_tucker_als = tucker_als(Y, R, 'tol', tol, 'maxiters', maxiters);

    normY = norm(Y);
    tucker_ts_error = norm(Y - tensor(ttensor(G_ts, A_ts)))/normY
    tucker_ttmts_error = norm(Y - tensor(ttensor(G_ttmts, A_ttmts)))/normY
    tucker_als_error = norm(Y - tensor(Y_tucker_als))/normY
    err_mat(1,i) = tucker_als_error;
    err_mat(2,i) = tucker_ts_error;
    err_mat(3,i) = tucker_ttmts_error;
end 
err_mat



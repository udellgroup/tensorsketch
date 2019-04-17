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

%R_true =  [10 10 10]; % True tensor rank
%R = [10 10 10]; % Algorithm target rank
R_true = [5 5 5]; % True tensor rank
R = [5 5 5]; % Algorithm target rank
I = [100 100 100]; 
%I = [400 400 400]; % Tensor size
K = 10; % Sketch dimension parameter
J1 = K*prod(R)/min(R); % First sketch dimension
J2 = K*prod(R); % Second sketch dimension
noise_level = 0.1; % Amount of noise added to nonzero elements
%tol = 1e-3; % Tolerance
%maxiters = 50; % Maximum number of iterations

%% Generate random dense tensor

fprintf('Generating dense tensor... ');
% G_true = tensor(randn(R_true));
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
true_signal_mag = norm(Y)^2 ;

Y = Y + randn(I)*noise_level*sqrt(noise_level^2*true_signal_mag/numel(Y)); 
fprintf('Done!\n\n');

fprintf('\n\nRunning tucker_als...\n')
tucker_als_tic = tic;
Y_tucker_als = tucker_als(Y, R, 'tol', tol, 'maxiters', maxiters);


normY = norm(Y);
tucker_als_error = norm(Y - tensor(Y_tucker_als))/normY

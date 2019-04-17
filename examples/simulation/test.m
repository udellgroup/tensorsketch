rerr_mat = zeros(3,10); 
Rs = 1:2:19;
I = ones(1,3)*200;

noise_level = 0.1 ; % Amount of noise added to nonzero elements
tol = 1e-3; % Tolerance
maxiters = 50; % Maximum
i = 1;
r = 5; 
%R_true =  [10 10 10]; % True tensor rank
%R = [10 10 10]; % Algorithm target rank
R_true = ones(1,3)*r; % True tensor rank
R = ones(1,3)*r; % Algorithm target rank
K = 2*r; % Sketch dimension parameter
J1 = K*prod(R)/min(R); % First sketch dimension
J2 = K*prod(R); % Second sketch dimension
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
true_signal_mag = norm(Y)^2; 
Y = Y + noise_level*randn(I)*sqrt(noise_level^2*true_signal_mag/numel(Y)); 
fprintf('Done!\n\n');

 %% Run tucker_als

fprintf('\n\nRunning tucker_als...\n')
tucker_als_tic = tic;
Y_tucker_als = tucker_als(Y, R, 'tol', tol, 'maxiters', maxiters);
tucker_als_toc = toc(tucker_als_tic);
%% Run tucker_ts

fprintf('\nRunning tucker_ts...\n\n')
tucker_ts_tic = tic;
[G_ts, A_ts] = tucker_ts(Y, R, J1, J2, 'tol', tol, 'maxiters', maxiters, 'verbose', false);
tucker_ts_toc = toc(tucker_ts_tic);

%% Run tucker_ttmts

fprintf('\nRunning tucker_ttmts...\n\n')
tucker_ttmts_tic = tic;
[G_ttmts, A_ttmts] = tucker_ttmts(Y, R, J1, J2, 'tol', tol, 'maxiters', maxiters, 'verbose', false);
tucker_ttmts_toc = toc(tucker_ttmts_tic);



%% Results

fprintf('\n\nComputing errors... ');
normY = norm(Y);
tucker_ts_error = norm(Y - tensor(ttensor(G_ts, A_ts)))/normY;
tucker_ttmts_error = norm(Y - tensor(ttensor(G_ttmts, A_ttmts)))/normY;
tucker_als_error = norm(Y - tensor(Y_tucker_als))/normY;
fprintf('Done!\n')

%% Put the result in the list 

rerr_mat(1,i) = tucker_als_error; 
rerr_mat(2,i) = tucker_ts_error; 
rerr_mat(3,i) = tucker_ttmts_error; 

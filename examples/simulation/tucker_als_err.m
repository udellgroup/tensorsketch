function err = tucker_als_err(Y, R, tol, maxiters)
    Y = tensor(Y);
    Y_tucker_als = tucker_als(Y, R, 'tol', tol, 'maxiters', maxiters);
    normY = norm(Y);
    err = norm(Y - tensor(Y_tucker_als))/normY;
end 




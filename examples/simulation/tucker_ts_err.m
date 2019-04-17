function err = tucker_ts_err(Y, R, J1, J2, tol, maxiters)
    Y = tensor(Y); 
    [G_ts, A_ts] = tucker_ts(Y, R, J1, J2, 'tol', tol, 'maxiters', maxiters); 
    normY = norm(Y);
    err = norm(Y - tensor(ttensor(G_ts, A_ts)))/normY;
end 

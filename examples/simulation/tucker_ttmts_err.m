function err = tucker_ttmts_err(Y, R, J1, J2, tol, maxiters)
    Y = tensor(Y); 
    [G_ttmts, A_ttmts] = tucker_ttmts(Y, R, J1, J2, 'tol', tol, 'maxiters', maxiters);
    normY = norm(Y);
    err = norm(Y - tensor(ttensor(G_ttmts, A_ttmts)))/normY;
end 


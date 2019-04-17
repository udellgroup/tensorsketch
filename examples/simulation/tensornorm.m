function val = tensornorm(T)
    Tv = reshape(T, numel(T),1); 
    val = norm(Tv);
end
function c = default_commutation(a,b)         
    c = a*b - b*a;
    if Config().simplify_expressions
        c = c.simplify();
    end
end
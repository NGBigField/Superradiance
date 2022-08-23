function a = A(i)
    arguments
        i (1,1) {mustBeInteger, mustBePositive}
    end
    Sym = CommonSymbols;
    a = ( ...
          S("+") * exp(  1i * Sym.w * Sym.t(i) ) ...
        + S("-") * exp( -1i * Sym.w * Sym.t(i) ) ...
        ) * (-1i/Sym.h_bar)*Sym.d * Sym.E(i) ;
end
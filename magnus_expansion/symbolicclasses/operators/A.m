function a = A(i)
    arguments
        i (1,1) {mustBeInteger, mustBePositive}
    end
    % Unpack symbols:
    sym_ = CommonSymbols();
    E_i     = sym_.E(i);
    t_i     = sym_.t(i);
    w       = sym_.w;
    d_h     = sym_.d_h;
    Exp_p_i = sym_.Exp_p(i);
    Exp_m_i = sym_.Exp_m(i);

    % Define A_i
    if Config().simplified_coefficients
        a = ( ...
            S("+") * exp(  1i * t_i )...
            + S("-") * exp( -1i * t_i )...
        ) * (-1i) ;    
    else
        a = ( ...
            S("+") * exp(  1i * w * t_i ) ...
            + S("-") * exp( -1i * w * t_i ) ...
        ) * (-1i*d_h) * E_i ;
    end
end
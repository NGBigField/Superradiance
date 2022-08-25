function obj = A(i)
    arguments
        i (1,1) {mustBeInteger, mustBePositive}
    end
    % Unpack symbols:
    sym_ = CommonSymbols();
    E_i   = sym_.E(i);
    t_i   = sym_.t(i);
    d     = sym_.d;
    w     = sym_.w;
    h_bar = sym_.h_bar;
    % Define obj
    obj = ( ...
          S("+") * exp(  1i * w * t_i ) ...
        + S("-") * exp( -1i * w * t_i ) ...
    ) * (-1i/h_bar)*d * E_i ;
end
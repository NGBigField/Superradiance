function Y = harmonicY(l,th,ph)
%HARMONICY  Spherical harmonic functions.
%
%   Y = HARMONICY(N,TH,PHI) computes the surface spherical harmonic of
%   degree N and orders M=0,..,N, evaluated at each element of inclination
%   TH and azimuth PHI. N and M must be scalar integers where M <= abs(N).
%   TH and PHI must be arrays of equal size.
%   Y(:,:,m+1) contains the m-th Harmonic

    Plm = legendre(l,cos(th)); %Associated Legendre functions
    
    Y = NaN([size(th),l+1]); %Initialize
    
    for m = 0:l
        a = (2*l+1)*factorial(l-m);
        b = 4*pi*factorial(l+m);
        C = sqrt(a/b);

        if l == 0
            Y(:,:,m+1) = C .*Plm .*exp(1i*m*ph);
        else
            Y(:,:,m+1) = C .*squeeze(Plm(m+1,:,:)) .*exp(1i*m*ph);
        end
    end
end
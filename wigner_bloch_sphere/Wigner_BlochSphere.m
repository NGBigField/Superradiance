function [W, X, Y, Z, TH, PH] = Wigner_BlochSphere(Npoints, N, psi, rho, statetype)
%Function that computes the atomic Wigner distribution on the Bloch sphere
%Based on https://doi.org/10.1103/PhysRevA.49.4101

% INPUTS
% Npoints   Number of points on the sphere
% N         Number of atoms
% psi       State wavefunction - relevant if statetype = 'psi'
% rho       State density matrix - relevant if statetype = 'rho'. Can be
%           sparse.
% statetype 'psi' or 'rho' depending on what initial state you want to feed
%
%
% OUTPUTS
% W         Wigner function
% X,Y,Z     Cartesian coordinates
% TH,PH     Polar coordinates

    [X,Y,Z] = sphere(Npoints);

    TH = acos(Z);
    PH = atan(Y./X); PH(X<0) = PH(X<0) + pi; PH(isnan(PH)) = pi;

    W = zeros(size(PH));
    j = N/2;
    for k = 0:2*j
        display_progress_bar(k+1,2*j+1)
        YKQ = harmonicY(k,TH,PH);
        for q = -k:k
            
            %Spherical Harmonic
            if q >= 0
                Ykq = YKQ(:,:,q+1);
            else
                Ykq = (-1)^q*conj(YKQ(:,:,-q+1));
            end

            %Compute Gkq
            Gkq = 0;
            for m1 = -j:j
                for m2 = -j:j

                    if -m1 + m2 + q == 0 %Selection rule for the Wigner symbol
                        switch statetype
                            case 'psi'
                                tracem1m2 = conj(psi(m1+j+1))*psi(m2+j+1);
                            case 'rho'
                                tracem1m2 = rho(m1+j+1,m2+j+1);
                            otherwise
                                error('Invalid statetype')
                        end
                        Gkq = Gkq + tracem1m2*sqrt(2*k+1)*(-1)^(j-m1)*conj(Wigner3j([j,k,j],[-m1,q,m2]));
                    end
                end
            end

            %Wigner
            W = W + Ykq*Gkq;
        end
    end

    if max(max(abs(imag(W)))) > 1e-3
        warning(['The wigner function has non negligible imaginary part ', num2str(max(max(abs(imag(W)))))]);
    end
    W = real(W);        %Remove spurious imaginary parts

    sph = surf(X,Y,Z,W);
    shading 'flat';
    axis equal;
    caxis(max(abs(caxis))*[-1,1]); %Make colorbar symmetric
    %colormap(mycolormap([0 0 0.4; 0 0 1; 0.9 0.89 0.35; 1 0 0; 0.4 0 0])); %Dark blue, blue, yellow, red, dark red
    colormap(mycolormap([0 0 0.4; 0 0 1; 0.95 0.95 0.95; 1 0 0; 0.4 0 0])); %Dark blue, blue, yellow, red, dark red
    %colorbar;

    %Plot the reference lines on the sphere
    hold on;
    t = linspace(0,2*pi,300);
    for ph = [0:pi/4:3/4*pi]
        plot3(sin(t)*cos(ph),sin(t)*sin(ph),cos(t),'Color',0.2*[1 1 1],'LineWidth',0.3,'LineStyle',':');
    end
    for th = pi*[1/2]
        plot3(sin(th)*sin(t),sin(th)*cos(t),cos(th)*ones(size(t)),'Color',0.2*[1 1 1],'LineWidth',0.3,'LineStyle',':'); %Equator
    end
    
    %Draw the arrows of the axes
    hold on;
    mArrow3([0,0,0], [0,0,1.4]);
    mArrow3([0,0,0], [0,1.4,0]);
    mArrow3([0,0,0], [1.4,0,0]);
    
    view([120, 30])
    %ax = gca; ax.Visible = 'off'; %Hide axes
    
    %Add lighting
    sph.FaceLighting = 'gouraud';
    [az,el] = view;                         %Get line of sight
    az = az/180*pi; el = el/180*pi;         %Convert to radiants
    [v1, v2, v3] = sph2cart(az-pi/2,el,1);  %direction of the light
    l = light('Position', [v1, v2, v3]);    %Add light
    material dull;                          %Make not too shiny
    
    hold off;
end
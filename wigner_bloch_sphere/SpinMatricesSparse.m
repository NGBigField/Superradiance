function [Sx,Sy,Sz] = SpinMatricesSparse(N)
%Function that finds the collective spin matrices in the symmetrized
%sector, in sparse form.
%
% INPUTS
% N         Number of particles
%
% OUTPUTS
% Sx,Sy,Sz  Collective matrices (sum of N paulis)
%
% CONVENTION
%There are N+1 elements in the basis. The convention we take is that the
%n-th element corresponds to n-1 particles in the ground state.
%The first element corresponds to the ground state (all spin down)
    
    n = [1:N];
    Sp = spdiags(2*sqrt(n.*(N-n+1))',-1,N+1,N+1);
    Sm = Sp';
    Sz = spdiags([-N:2:N]',0,N+1,N+1);

    Sx = (Sp + Sm)/2;
    Sy = (Sp - Sm)/(2*1i);
end
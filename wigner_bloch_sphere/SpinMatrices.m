function [Sx,Sy,Sz] = SpinMatrices(N)
%Function that finds the collective spin matrices in the symmetrized
%sector.
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

    %s = N/2; %Total spin quantum number
    %n = [1:N];
    %Sp = diag(sqrt(n.*(N-n+1)),-1);
    %n = [2:N+1];
    %Sm = diag(sqrt((N-n+2).*(n-1)),+1);
    %Sz = diag([-s:s]);
    
    n = [1:N];
    Sp = diag(2*sqrt(n.*(N-n+1)),-1);
    n = [2:N+1];
    Sm = diag(2*sqrt((N-n+2).*(n-1)),+1);
    Sz = diag([-N:2:N]);

    Sx = (Sp + Sm)/2;
    Sy = (Sp - Sm)/(2*1i);
end
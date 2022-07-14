%% Clear:
close all; clear; clc;

%% Init params:
Npoints = 800;
N = 10;
psi = [0.9, 0.1, 0, 0,  0, 0, 0, 0, 0, 0, 3];
rho = [];
statetype = 'psi';

%% call:
[W, X, Y, Z, TH, PH] = Wigner_BlochSphere(Npoints, N, psi, rho, statetype);
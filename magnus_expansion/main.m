clear all; close all; clc;
addpath(genpath(pwd));
s_ = CommonSymbols();
%
expo = exp(1i * 3*s_.t(1));
s = S("+");
%%

disp("A1=");
disp(~A(1));
disp("[A1,A2]:=");
disp(~[A(1), A(2)]);
disp("[A1, [A2,A3]] + [A3, [A2,A1]]:"); 
disp(~( [A(1), [A(2), A(3)] ] + [A(3), [A(2), A(1)] ] ));


%%
item4 = [[[A(1),A(2)], A(3)], A(4)] ... 
    + [A(1), [[A(2), A(3)], A(4)]]  ...
    + [A(1), [A(2), [A(3), A(4)]]]  ...
    + [A(2), [A(3), [A(4), A(1)]]] 


%%

% %%
% [r,sigma] = subexpr( ~res3  ) 
% %%
% syms N k n m
% pi = sym(pi);
% f = exp(1j * 2*pi * k * (n-m)/N  );
% a = symsum(f,k,0,N-1)
% %%
% a_new = subs(a,     n, 0 )
% a_new = subs(a_new, N, 10)
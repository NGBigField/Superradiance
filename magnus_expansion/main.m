clear all; close all; clc;
addpath(genpath(pwd));

%%
syms_ = CommonSymbols();

disp("A1=");
disp(~A(1));
disp("[A1,A2]:=");
disp(~[A(1), A(2)]);
disp("[A1, [A2,A3]] + [A3, [A2,A1]]:"); 
res3 = [A(1), [A(2), A(3)] ] + [A(3), [A(2), A(1)] ] ;
expression = ~res3;
%%
[r,sigma] = subexpr( ~res3  ) 
%%
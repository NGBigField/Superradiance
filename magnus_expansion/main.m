clear all; close all; clc;
% reset(symengine)

% mute::
%#ok<*OR2> 

%%
s_ = CommonSymbols;

%%
ss = SS( S("+")*5 , S("-")*4 );
s1 = ss.get(1);
s2 = ss.get(2);
similar = s1 | s2;
if similar
    disp(3)
end

A1 = A(1);
A2 = A(2);

res = [A1, A2]
%%


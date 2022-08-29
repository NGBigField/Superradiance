%% unpack common symbols:
s = CommonSymbols();

%% i = 0:
i=2;
odd_item = A(1);
even_item = [A(1),A(2)]; 
[x, y, z] = split_xyz(even_item, odd_item);

%% int
syms T
X = int(x, s.t(1), 0, T )
Y = int(y, s.t(1), 0, T )
Z = int(z, s.t(1), 0, inf )
Z = int(Z, s.t(2), 0, inf )

%% Subs: 
function [x,y,z] = split_xyz(odd, even)
    arguments
        odd  (1,1) BaseSymbolicClass
        even (1,1) BaseSymbolicClass
    end
    % Get basic results:
    [x_o, y_o, z_o] = odd.xyz();
    [x_e, y_e, z_e] = even.xyz();    
    % Assert values that should be 0:
    for expr = [x_o, y_o, z_e]
        assert( isAlways(expr==0) )
    end
    % Return values:
    x = x_e;
    y = y_e;
    z = z_o;
end

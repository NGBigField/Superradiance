classdef A < Sum

    
    properties
        i (1,1) {mustBeInteger}
    end
    
    methods
        function obj = A(i)
            arguments
                i (1,1) {mustBeInteger}
            end            
            [Sp, Sm] = create_subs(i);
            obj@Sum(Sp,Sm);
            obj.i = i; 
            obj.coef = (-1i/obj.global_symbols.h_bar)*obj.global_symbols.d*obj.E;
        end       

        function res = E(obj)
            res = obj.global_symbols.E(obj.i);
        end


        
    end % methods
end

%% Supress messages:
%#ok<*PROPLC> 
%#ok<*PROP> 

function [Sp, Sm] = create_subs(i)
    Sp = S('+') ;
    Sp.coef = ~Exp(i,'+');
    Sm = S("-") ;
    Sm.coef = ~Exp(i,'-');
end
classdef Product < SuperOperator

    properties (Constant)
        connector  = "*"
        base_value = 1
        base_op    = UnitOperator()
    end
   
    methods (Static)
        function res = connect(A,B)
            res = A*B;
        end    
    end
    methods
        function res = multiply(A,B)
            arguments
                A (1,1) Sum
                B (1,1) Sum
            end
            error("SymbolicClass:NotImplemented","Not implemented");    
        end

    end % methods
end

%% Supress messages:
%#ok<*PROPLC> 
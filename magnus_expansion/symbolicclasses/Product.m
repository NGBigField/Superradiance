classdef Product < SuperOperator

    properties (Constant, Hidden)
        connector  = "*"
        neutral_op = UnitOperator()
    end
   
    methods (Static)
        function res = connect(A,B)
            res = A*B;
        end    
    end
    methods
        function obj = Product(operators)
            arguments (Repeating)
                operators (1,1) BaseSymbolicClass
            end
            % Derive total coef and strip ops from their coefficients:
            coef = 1;
            for i = 1 : length(operators)
                coef = coef * operators{i}.coef;
                operators{i}.coef = 1;
            end            
            % apply super constructor:
            obj@SuperOperator(operators{:});
            obj.coef = coef;            
        end
        %%
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
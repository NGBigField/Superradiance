classdef SuperOperator < BaseSymbolicClass

    properties
        subs (:,1) cell
    end

    properties (Abstract, Constant, Hidden)
        connector  (1,1)  %  * or +
        neutral_op (1,1) BaseSymbolicClass        
    end

    methods 
        function obj = SuperOperator(operators)
            arguments (Repeating)
                operators (1,1) BaseSymbolicClass
            end
            obj.subs = operators;
        end
        %%
        function sym_expression = unpack(obj)
            sym_expression = ~obj.neutral_op;            
            for i = 1 : length(obj.subs)
                op = obj.subs{i};
                sym_expression = obj.connect(sym_expression, ~op);
            end
            sym_expression = sym_expression * obj.coef;
        end
        %%
        function res = num_subs(obj)
            res = length(obj.subs);
        end
    end


    methods (Abstract, Static)
        res = connect(A,B)  % join expressions
    end

end % classdef
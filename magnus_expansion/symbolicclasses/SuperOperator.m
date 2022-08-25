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
        function super_express = unpack(obj)
            % initial value:
            super_express = ~obj.neutral_op;            
            % accumulate sub operators:
            for i = 1 : length(obj.subs)
                op = obj.subs{i};
                op_express = ~op;
                super_express = obj.connect(super_express, op_express);
            end
            % Apply global coef (if applicable)
            super_express = super_express * obj.coef;
            % Simplify:
            if Config().simplify
                super_express = simplify(super_express);
            end
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
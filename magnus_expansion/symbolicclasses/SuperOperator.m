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
        function obj = simplify(obj)        
            % simplify global coef:
            try
                obj.coef = BaseSymbolicClass.simplify_coef(obj.coef);
            catch ME
                if string(ME.identifier) ~= "SymbolicClass:LockedProperty"  % This is expected in Sum object
                    rethrow(ME)
                end
            end
            
            % Simplify each of the inner operators:
            for i = 1 : obj.num_subs
               obj.subs{i} = obj.subs{i}.simplify();
            end

        end
        %%
        function super_express = unpack(obj)
            % Simplify:
            if Config().simplify
                obj = obj.simplify();
            end
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
            % Simplify again:
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
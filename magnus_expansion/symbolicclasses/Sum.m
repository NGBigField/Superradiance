classdef Sum < SuperOperator

    properties (Constant)
        connector  = "+"
        base_value = 0
        base_op    = ZeroOperator()
    end

    methods (Static)
        function res = connect(A,B)
            res = A+B;
        end
    end
    methods      
        function obj = Sum(operators)           
            arguments (Repeating)
                operators (1,1) BaseSymbolicClass
            end
            for i = 1 : length(operators)
                op = operators{i};
                obj.reducing_similars_terms_or_add(op);
            end
        end

        function [] = reducing_similars_terms_or_add(obj, op_in)
            arguments
                obj   (1,1) Sum
                op_in (1,1) BaseSymbolicClass
            end
            for i = 1 : length(obj.subs)
                op_sub = obj.subs{i};
                if op_sub == op_in
                    op_reduced = op_sub + op_in;
                    obj.subs{i} = op_reduced;  % Replace
                    return
                end
            end
            obj.subs{end+1,1} = op_in;
        end

        function res = multiply(A,B)
            arguments
                A (1,1) Sum
                B (1,1) Sum
            end
            % Check type:         
            
            % Iterate inner types to deduce final expression
            cumulative = Sum.base_op;
            operators = {};
            for i = 1 : length(A.subs)
                a = A.subs{i};
                for j = 1 : length(B.subs)
                    b = B.subs{j};
                    
                    % new operator
                    crnt = a*b;
                    cumulative = cumulative + crnt ;

                end % for j
            end % for i

        end

        
    end % methods
end

%% Supress messages:
%#ok<*PROPLC> 
classdef Sum < SuperOperator

    properties (Constant, Hidden)
        connector  = "+"
        neutral_op = ZeroOperator()
    end

    methods (Static)
        function res = connect(A,B)
            res = A+B;
        end    
    end
    methods      
        %%
        function obj = Sum(operators_in)           
            arguments (Repeating)
                operators_in (1,1) BaseSymbolicClass
            end
            % Construct sum part by part, allow reduction of similiar expressions
            for i = 1 : length(operators_in)
                op = operators_in{i};
                if isa(op,'Sum')
                    ops = op.subs;
                    for j = 1 : length(ops)
                        op = ops{j};
                        obj.reducing_similars_terms_or_add(op);
                    end
                elseif isa(op,'BaseSymbolicClass')
                    obj.reducing_similars_terms_or_add(op);
                else
                    error("SymbolicClass:UnsupportedCase","Not a legit case");  
                end
            end
        end
        %%
        function res = simmilar(obj, other)
            error("SymbolicClass:NotImplemented", "This function is not implemented");
        end
        %%
        function [] = reducing_similars_terms_or_add(obj, op_in)
            arguments
                obj   (1,1) Sum
                op_in (1,1) BaseSymbolicClass
            end
            for i = 1 : length(obj.subs)
                op_sub = obj.subs{i};
                similar_operators = op_sub | op_in ; 
                if similar_operators
                    op_reduced = op_sub / op_in ; % reduce
                    obj.subs{i} = op_reduced;  % Replace
                    return
                end
            end
            obj.subs{end+1,1} = op_in;
        end
        %%
        function res = multiply(a,b)
            arguments
                a (1,1) 
                b (1,1) 
            end
            % Check type:         
            if isa(b, 'Sum')
                res = Sum.multiply_sum_by_sum(a,b);
            elseif isa(b, 'sym')
                res = Sum.multiply_sum_by_coef(a,b);
            else
                error("SymbolicClass:UnsupportedCase","Not a legit case");    
            end

        end
        %%
    end % methods

    methods (Static)
        function res = multiply_sum_by_sum(sum_a,sum_b)
            arguments
                sum_a (1,1) Sum
                sum_b (1,1) Sum
            end
            res = Sum.neutral_op;
            for i = 1 : sum_a.num_subs()
                a = sum_a.subs{i};
                for j = 1 : sum_b.num_subs()
                    b = sum_b.subs{j};
                    
                    % new operator
                    crnt = a*b;
                    res = res + crnt ;

                end % for j
            end % for i            
        end
        %%
        function s = multiply_sum_by_coef(s, c)
            arguments
                s (1,1) Sum
                c (1,1) sym
            end
            for i = 1 : s.num_subs()
                s.subs{i} = s.subs{i} * c;
            end
        end
    end % static methods
    
    %% Getters and Setters:
    methods (Static)
        function val  = set_coef_validation(val)
            error("SymbolicClass:LockedProperty", "Changing `coef` is not supported for class `Sum`");
        end
    end

end

%% Supress messages:
%#ok<*PROPLC> 
%#ok<*BDSCA> 
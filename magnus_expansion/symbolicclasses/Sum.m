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
                        obj = obj.reduce_similars_terms_or_add(op);
                    end
                elseif isa(op,'BaseSymbolicClass')
                    obj = obj.reduce_similars_terms_or_add(op);
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
        function [obj] = reduce_similars_terms_or_add(obj, op_in)
            arguments
                obj   (1,1) Sum
                op_in (1,1) BaseSymbolicClass
            end
            % Assume:
            replacement_occurred = false;
            % Go over all existing sub-operators:
            for i = 1 : length(obj.subs)
                op_sub = obj.subs{i};
                % Check similiarity between operators:
                similar_operators = op_sub | op_in ; 
                if similar_operators
                    % Reduce
                    reduced = op_sub / op_in ; 
                    if iscell(reduced) % Meaning that we got a replacement and a residue
                        replacement = reduced{1};
                        residue = reduced{2};
                    else
                        replacement = reduced;
                        residue = [];
                    end
                    % Execute replacement of old op:
                    obj.subs{i} = replacement;  % Replace
                    replacement_occurred = true;
                    % Manage residue, if exists
                    if ~isempty(residue)
                        obj = obj.reduce_similars_terms_or_add(residue);
                    end
                    break
                end
            end
            % If didn't find any similar, just add it:
            if ~replacement_occurred
                obj.subs{end+1,1} = op_in;
            end
            % perform clean-up:
            obj = obj.cleanup();
        end
        %%
        function obj = cleanup(obj)
            % Find zero operators:
            zero_indices = [];
            for i = 1 : obj.num_subs()
                op = obj.subs{i};
                if isa(op, 'ZeroOperator')
                    zero_indices = [zero_indices, i];
                end
            end            
            % remove zero operators:
            if isempty(zero_indices) 
                return
            end
            obj.subs(zero_indices) = [];
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
            elseif isa(b, 'sym') || isnumeric(b)
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
                c (1,1) 
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
classdef (Abstract) BaseSymbolicClass < handle
    
    properties (Constant, Hidden)
        global_symbols (1,1) CommonSymbols = CommonSymbols()
    end
    %%
    properties
        coef (1,1) = 1  % can be symbolic
    end
    %%
    methods (Abstract)
        res = unpack(obj)
        res = multiply(obj, other)
        res = simmilar(obj, other)
    end
    %%
    methods 
        function c = commutations(a, b)
            c = a*b - b*a;
        end
        function res = add_ops(A,B)
            res = Sum(A,B);
            % if a reduction of expressions occurred. return it the single op (instead of a Sum of ops)
            if res.num_subs == 1
                res = res.subs{1};
            end
        end
        function res = reduce(a,b)
            % assert reducable
            assert( a.simmilar(b) );
            % reduce:
            new = a;
            new.coef = a.coef + b.coef;
            if new.coef == 0
                res = 0;
            else
                res = new;
            end
        end
    end

    %% Override [ , ]   * and ~  operators:
    methods
        function res = not(obj)  % Override ~A 
            res = obj.unpack();
        end
        function res = horzcat(A,B)  % Override [A,B] 
            res = A.commutations(B);
        end
        function res = mtimes(A,B)  % Override A*B 
            res = A.multiply(B);
        end
        function res = plus(A,B)  % Override A + B 
            res = add_ops(A,B);
        end 
        function res = or(A,B)  % override A|B 
            res = A.simmilar(B);
        end
        function res = minus(A,B)  % Override A - B  (not the space is obligated)
            res = reduce(A,B);
        end 
    end
end


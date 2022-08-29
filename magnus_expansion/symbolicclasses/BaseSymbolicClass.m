classdef (Abstract, InferiorClasses={?sym} ) BaseSymbolicClass
        
    properties
        coef (1,1) = 1   % can be symbolic
        expression (1,1)  % used to for default `repr` of class, via the getter methods
    end
    %%
    methods (Abstract)
        res = unpack(obj)
        res = multiply(obj, other)
        res = simmilar(obj, other)
    end
    %%
    methods 
        function a = simplify(a)
            a.coef = BaseSymbolicClass.simplify_coef( a.coef );
            if a.coef == 0
                a = ZeroOperator;
            end
        end
        function [x,y,z] = xyz(obj)
            % Classes which support splitting operator into x,y,z should implement this method.
            error("SymbolicClass:IncompatibleMethod", "This method does not exist for class '"+string(class(obj))+"'."); 
        end
        function c = commutations(a, b)
            c = operators.default_commutation(a,b);
        end
        function res = add_ops(A,B)
            res = Sum(A,B);
            % if a reduction of expressions occurred. return it the single op (instead of a Sum of ops)
            if res.num_subs == 1
                assert(res.coef==1)
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
                res = ZeroOperator;
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
        function res = minus(A,B)  % Override A - B 
            res = add_ops(A,B*(-1));
        end 
        function res = or(A,B)  % override A | B 
            res = A.simmilar(B);
        end
        function res = mrdivide(A,B)  % Override A / B  (not the space is obligated)
            res = reduce(A,B);
        end 
    end
    %% Setters and Getters:
    methods  % Getter and Setters:
        function obj = set.coef(obj,val)
            obj.coef = obj.set_coef_validation(val);
        end
        function val = get.coef(obj)
            val = obj.coef;
        end
        function val = get.expression(obj)
            val = ~obj;
        end
        function set.expression(obj,val)
            error("SymbolicClass:LockedProperty", "Expression can't be changed explicitly.");
        end        
    end % getters and setters
    %% Static Operators
    methods (Static)
        % Used when locked property is needed
        function val = set_coef_validation(val)
            % Do Nothing
        end
        
        function coef = simplify_coef(coef)
            if isa(coef,'sym')
                coef = simplify(coef);
            end
        end
        function res = is_zero(a)
            if isa(a, 'ZeroOperator')
                res = true;
            elseif isa(a, 'BaseSymbolicClass') && a.coef==0
                res = true;
            elseif isnumeric(a) && length(a)==1 && a==0
                res = true;
            else
                res = false;
            end
            
        end
    end
end


%% Helper functions:

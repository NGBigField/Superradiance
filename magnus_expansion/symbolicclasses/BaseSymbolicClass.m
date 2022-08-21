classdef (Abstract) BaseSymbolicClass
    
    properties (Constant)
        global_symbols (1,1) CommonSymbols = CommonSymbols()
    end
    
    properties
        coef           (1,1) = 1  % can be symbolic
    end
    
    methods (Abstract)
        res = unpack(obj)
        res = multiply(obj, other)
    end

    methods 
        function C = commutations(A, B)
            C = A*B - B*A;
        end
    end

    %% Override [ , ]   * and ~  operators:
    methods
        function res = not(obj)  %  Override   ~A  operation
            res = obj.unpack();
        end

        function res = horzcat(A,B) %  Override  [A , B]  operation
            res = A.commutations(B);
        end

        function res = mtimes(A,B) %  Override  A*B  operation
            res = A.multiply(B);
        end

        function res = plus(A,B) % Override A+B operation
            res = Sum(A,B);
        end 
    end
end


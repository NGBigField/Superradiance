classdef ZeroOperator < BaseSymbolicClass


    methods 
        function obj = ZeroOperator()
            obj.coef = 0;
        end
        function res = unpack(obj)
            res = 0;
        end
        function res = multiply(obj, other)
            res = ZeroOperator();
        end
        function res = simmilar(obj, other)
            res = true;
        end
        function res = reduce(obj, other)
            res = other;
        end
    end
end
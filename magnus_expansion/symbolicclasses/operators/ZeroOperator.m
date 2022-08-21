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
    end
end
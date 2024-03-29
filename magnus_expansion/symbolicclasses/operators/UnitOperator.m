classdef UnitOperator < BaseSymbolicClass

    methods 
        function obj = UnitOperator()
            obj.coef = 1;
        end
        function res = unpack(obj)
            res = 1;
        end
        function res = multiply(obj, other)
            res = other();
        end
        function res = simmilar(obj, other)
            error("SymbolicClass:NotImplemented", "This function is not implemented");
        end
    end
end
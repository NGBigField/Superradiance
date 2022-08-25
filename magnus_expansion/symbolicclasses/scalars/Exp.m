classdef Exp < BaseSymbolicClass


    properties
        i    (1,1) {mustBeInteger}
        sign (1,1) {mustBeNumeric}
    end
    
    methods
        function obj = Exp(i,sign)
            arguments
                i     (1,1) {mustBeInteger}
                sign  (1,1) {mustBeMember(sign,[+1,-1, "+", "-"])}
            end
            obj.i = i;       
            obj.sign = parse_sign(sign);
        end       

        function [out] = unpack(obj)
            % Parse symbolic common expressions:
            w = obj.global_symbols.w;
            t = obj.global_symbols.t(obj.i);
            pm = obj.sign; % plus or minus

            % output: 
            out = exp( pm * 1i * w * t);
        end

        function C = commutations(A,B)
            error("Not implemented");
        end
        function res = multiply(A,B)
            error("Not implemented");
        end

    end
end


function pm = parse_sign(sign)
    switch sign
        case {"+", 1}
            pm = 1;
        case {"-", -1}
            pm = -1;
        otherwise
            error("SymbolicClass:UnsupportedCase","Not a legit case");    
    end
end
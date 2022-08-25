classdef S < BaseSymbolicClass

    properties (SetAccess=immutable)
        script (1,1) string % {mustBeMember(script,["z","p","m"])}         
    end
    properties (Constant, Hidden)
        standard_order (1,:) = ["z", "+", "-"];
    end

    methods
        %%
        function obj = S(script)
            arguments
                script (1,1) string {mustBeMember(script,["z","+","-"])}
            end
            obj.script = standard_script(script);
        end       
        %%
        function res = unpack(obj)
            syms_ = CommonSymbols;
            switch obj.script
                case "z"
                    res = syms_.Sz;
                case "+"
                    res = syms_.Sp;
                case "-"
                    res = syms_.Sm;
                otherwise
                    error("SymbolicClass:UnsupportedCase","Not a legit case");    
            end
            res = res * obj.coef;
        end
        %%
        function res = multiply(obj, other)
            if isa(other, 'S')
                res = S.multiply_by_s(obj, other);
            elseif isa(other, 'sym') || isa(other, 'numeric')
                res = S.multiply_by_coef(obj, other);
            elseif isa(other, 'Sum')
                res = Sum.multiply_by_operator(other, obj, "op_from","Left");
            else
                error("SymbolicClass:UnsupportedCase","Not a legit case");    
            end
        end
        %%
        function res = simmilar(obj, other)
            if isa(other, 'S') && other.is( obj.script )
                res = true;
            else
                res = false;
            end
        end
        %%
        function res = commutations(s1, s2)
            arguments
                s1 (1,1) S 
                s2 (1,1) S
            end
            if     s1.is("z") && s2.is("+"), res = S("+")*(+1);
            elseif s1.is("z") && s2.is("-"), res = S("-")*(-1);
            elseif s1.is("+") && s2.is("-"), res = S("z")*(+2);
            elseif s1.is("+") && s2.is("z"), res = S("+")*(-1);
            elseif s1.is("-") && s2.is("z"), res = S("-")*(+1);
            elseif s1.is("-") && s2.is("+"), res = S("z")*(-2);                
            else
                error("SymbolicClass:UnsupportedCase","Not a legit case");    
            end
        end
        %%
        function tf = is(obj, target_script)
            arguments
                obj
                target_script (1,1) string {mustBeMember(target_script,["z","+","-"])}
            end
            tf = standard_script(target_script)==obj.script;
        end
        %%
    end % methods

    methods (Static)
                %%
        function res = multiply_by_s(s1, s2)
            arguments
                s1 (1,1) S
                s2 (1,1) S
            end
            res = SPair(s1,s2);
        end
        %%
        function res = multiply_by_coef(s, c)
            arguments
                s (1,1) S
                c (1,1) sym
            end
            s.coef = s.coef * c;
            res = s;
        end
    end
end




function out = standard_script(script)
    switch script
        case {"Z","z"}
            out = "z";
        case {"+","P", "p"}
            out = "+";
        case {"-","M", "m"}
            out = "-";
        otherwise
            error("SymbolicClass:UnsupportedCase","Not a legit case");    
    end
end